"""Typer CLI entry points for the W&B monitoring agent."""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from wandb_agent.agent import MonitorAgent
from wandb_agent.approval import app as approval_app, init_app
from wandb_agent.config import AgentConfig
from wandb_agent.executor import ActionExecutor
from wandb_agent.poller import RunSnapshot, WandbPoller
from wandb_agent.store import RunStore

app = typer.Typer(name="wandb-agent", help="W&B AI Monitoring Agent", add_completion=False)

_CONFIG_PATH = Path.home() / ".wandb-agent" / "config.yaml"
_EXAMPLE_CONFIG = Path(__file__).parent.parent / "config.example.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> AgentConfig:
    if not _CONFIG_PATH.exists():
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _EXAMPLE_CONFIG.exists():
            shutil.copy(_EXAMPLE_CONFIG, _CONFIG_PATH)
            typer.echo(f"Created config at {_CONFIG_PATH} from example — please edit it.")
        else:
            typer.echo(f"Config not found at {_CONFIG_PATH}. Please create it.")
        raise typer.Exit(1)
    return AgentConfig.from_yaml(_CONFIG_PATH)


def _run_approval_server(store: RunStore, port: int) -> None:
    init_app(store)
    uvicorn.run(approval_app, host="0.0.0.0", port=port, log_level="warning")


def _process_approved_relaunches(
    store: RunStore, executor: ActionExecutor
) -> None:
    """Check for approved stop_and_relaunch diagnoses and execute them."""
    for diag in store.get_approved_stop_and_relaunch():
        run_info = store.get_run_info(diag.run_id)
        if not run_info:
            logger.warning("No run info for %s; cannot relaunch", diag.run_id)
            continue
        config = json.loads(run_info["config_json"])
        # Reconstruct a minimal snapshot sufficient for relaunch
        snapshot = RunSnapshot(
            run_id=run_info["run_id"],
            run_name=run_info["run_id"],
            project=run_info["project"],
            entity=run_info["entity"],
            state="running",
            config=config,
            history=[],
            system_metrics={},
            tags=[],
            slurm_job_id=None,
            snapshot_at=datetime.utcnow(),
        )
        executor.execute_approved_relaunch(diag, snapshot)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def monitor() -> None:
    """Start the polling loop (blocks until Ctrl-C)."""
    config = _load_config()

    if not config.ollama_model and not config.groq_model and not config.anthropic_api_key:
        typer.echo(
            "ERROR: No LLM configured. Set ollama_model, groq_model, or anthropic_api_key in config.",
            err=True,
        )
        raise typer.Exit(1)

    store = RunStore()

    # Start approval server in a background daemon thread
    server_thread = threading.Thread(
        target=_run_approval_server,
        args=(store, config.approval_server_port),
        daemon=True,
        name="approval-server",
    )
    server_thread.start()

    # Optionally set W&B API key from config
    if config.wandb_api_key:
        import os  # noqa: PLC0415

        os.environ.setdefault("WANDB_API_KEY", config.wandb_api_key)

    project_names = [p.name for p in config.projects]
    poll_interval = config.projects[0].poll_interval_s if config.projects else 45

    # Load training scripts per project (if configured)
    script_contents: dict[str, str] = {}
    for p in config.projects:
        if p.training_script_path:
            script_path = Path(p.training_script_path).expanduser()
            if script_path.exists():
                script_contents[p.name] = script_path.read_text()
                logger.info("Loaded training script for '%s' from %s", p.name, script_path)
            else:
                logger.warning("Training script not found for '%s': %s", p.name, script_path)

    poller = WandbPoller(entity=config.entity, projects=project_names, poll_interval_s=poll_interval)
    agent = MonitorAgent(
        api_key=config.anthropic_api_key,
        ollama_model=config.ollama_model,
        ollama_base_url=config.ollama_base_url,
        groq_api_key=config.groq_api_key,
        groq_model=config.groq_model,
    )
    executor = ActionExecutor(config=config, store=store)
    project_contexts: dict[str, str] = {}

    if config.ollama_model:
        llm_info = f"ollama / {config.ollama_model}"
    elif config.groq_model:
        llm_info = f"groq / {config.groq_model}"
    else:
        llm_info = f"anthropic / {agent.model}"

    typer.echo(
        f"Monitoring {len(project_names)} project(s) for entity '{config.entity}'. "
        f"LLM: {llm_info}. "
        f"Approval server on port {config.approval_server_port}. Ctrl-C to stop."
    )

    try:
        while True:
            # Process any previously approved relaunches
            _process_approved_relaunches(store, executor)

            # Poll W&B for running runs
            snapshots = poller.poll()

            for snapshot in snapshots:
                if snapshot.project not in project_contexts:
                    project_contexts[snapshot.project] = agent.ensure_context(
                        snapshot, script_contents.get(snapshot.project, "")
                    )

                past = store.get_past_diagnoses(snapshot.run_id)
                diagnosis = agent.diagnose(snapshot, past, context=project_contexts[snapshot.project])

                store.save_snapshot(snapshot)
                store.save_diagnosis(diagnosis)

                ts = datetime.now().strftime("%H:%M:%S")
                if diagnosis.status == "ok":
                    print(
                        f"[{ts}] [{snapshot.run_name}] OK ({diagnosis.confidence:.0%}) — "
                        f"{diagnosis.reasoning}"
                    )
                else:
                    print(
                        f"[{ts}] [{snapshot.run_name}] {diagnosis.status.upper()}: "
                        f"{diagnosis.failure_mode} ({diagnosis.confidence:.0%}) — "
                        f"{diagnosis.reasoning}"
                    )

                if diagnosis.status != "ok":
                    executor.execute(diagnosis, snapshot)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        typer.echo("\nShutting down cleanly.")


@app.command()
def approve(diagnosis_id: str) -> None:
    """Approve a pending stop_and_relaunch diagnosis."""
    store = RunStore()
    store.update_approval(diagnosis_id, approved=True, reason=None)
    typer.echo(f"Approved: {diagnosis_id}")


@app.command()
def reject(
    diagnosis_id: str,
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Rejection reason"),
) -> None:
    """Reject a pending stop_and_relaunch diagnosis."""
    store = RunStore()
    store.update_approval(diagnosis_id, approved=False, reason=reason)
    typer.echo(f"Rejected: {diagnosis_id}")


@app.command()
def pending() -> None:
    """List pending (unapproved) stop_and_relaunch diagnoses."""
    store = RunStore()
    diagnoses = store.get_pending_diagnoses()
    if not diagnoses:
        typer.echo("No pending diagnoses.")
        return
    for d in diagnoses:
        typer.echo(
            f"{d.diagnosis_id}  run={d.run_id}  failure={d.failure_mode}  "
            f"confidence={d.confidence:.0%}  ts={d.timestamp.strftime('%Y-%m-%d %H:%M')}"
        )


@app.command()
def status() -> None:
    """Show all tracked runs and their most recent diagnosis."""
    store = RunStore()
    run_ids = store.get_all_run_ids()
    if not run_ids:
        typer.echo("No runs tracked yet.")
        return
    for run_id in run_ids:
        diagnoses = store.get_past_diagnoses(run_id, limit=1)
        if diagnoses:
            d = diagnoses[0]
            typer.echo(
                f"{run_id}  status={d.status}  failure={d.failure_mode}  "
                f"confidence={d.confidence:.0%}  {d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )


if __name__ == "__main__":
    app()
