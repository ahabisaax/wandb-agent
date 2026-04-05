"""Action executor — notify, patch config, stop and relaunch."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import httpx

from wandb_agent.config import AgentConfig
from wandb_agent.poller import Diagnosis, RunSnapshot
from wandb_agent.store import RunStore

logger = logging.getLogger(__name__)

_PATCHES_DIR = Path.home() / ".wandb-agent" / "patches"


class ActionExecutor:
    def __init__(self, config: AgentConfig, store: RunStore) -> None:
        self.config = config
        self.store = store

    def execute(self, diagnosis: Diagnosis, snapshot: RunSnapshot) -> None:
        """Route to the appropriate action based on diagnosis.suggested_action."""
        action = diagnosis.suggested_action
        if action == "notify":
            self._notify(diagnosis, snapshot)
        elif action in ("patch_config", "stop_and_relaunch"):
            # All config changes and relaunches require explicit user approval
            self._notify_pending_approval(diagnosis, snapshot)

    # ------------------------------------------------------------------
    # Notify
    # ------------------------------------------------------------------

    def _notify(self, diagnosis: Diagnosis, snapshot: RunSnapshot) -> None:
        webhook = self.config.notify_slack_webhook
        diff_str = str(diagnosis.suggested_diff) if diagnosis.suggested_diff else "{}"
        message = (
            f"*[W&B Agent]* :warning: Run `{snapshot.run_name}` — "
            f"{diagnosis.failure_mode} ({diagnosis.confidence:.0%} confidence)\n"
            f"{diagnosis.reasoning}\n"
            f"Suggested diff: {diff_str}"
        )
        if not webhook:
            logger.info("Slack not configured. Notification: %s", message)
            return
        self._slack_post(webhook, message)

    def _notify_pending_approval(
        self, diagnosis: Diagnosis, snapshot: RunSnapshot
    ) -> None:
        webhook = self.config.notify_slack_webhook
        diff_str = str(diagnosis.suggested_diff) if diagnosis.suggested_diff else "{}"
        message = (
            f"*[W&B Agent]* :rotating_light: Run `{snapshot.run_name}` — "
            f"{diagnosis.failure_mode} ({diagnosis.confidence:.0%} confidence)\n"
            f"{diagnosis.reasoning}\n"
            f"Suggested diff: {diff_str}\n"
            f"Approve relaunch: `wandb-agent approve {diagnosis.diagnosis_id}`\n"
            f"Reject: `wandb-agent reject {diagnosis.diagnosis_id}`"
        )
        if not webhook:
            logger.info("Pending approval for diagnosis %s: %s", diagnosis.diagnosis_id, message)
            return
        self._slack_post(webhook, message)

    @staticmethod
    def _slack_post(webhook: str, text: str) -> None:
        try:
            httpx.post(webhook, json={"text": text}, timeout=10)
        except Exception as exc:
            logger.error("Failed to send Slack notification: %s", exc)

    # ------------------------------------------------------------------
    # Patch config
    # ------------------------------------------------------------------

    def _patch_config(self, diagnosis: Diagnosis, snapshot: RunSnapshot) -> Path:
        import yaml  # noqa: PLC0415

        _PATCHES_DIR.mkdir(parents=True, exist_ok=True)
        patch_path = _PATCHES_DIR / f"{diagnosis.diagnosis_id}.yaml"

        merged = dict(snapshot.config)
        for key, diff in diagnosis.suggested_diff.items():
            merged[key] = diff["after"] if isinstance(diff, dict) and "after" in diff else diff

        with open(patch_path, "w") as f:
            yaml.dump(merged, f)

        logger.info("Patch config written to %s", patch_path)
        print(f"Patch written to: {patch_path}")
        return patch_path

    # ------------------------------------------------------------------
    # Stop and relaunch (only called after approval)
    # ------------------------------------------------------------------

    def execute_approved_relaunch(
        self, diagnosis: Diagnosis, snapshot: RunSnapshot
    ) -> None:
        """Execute a previously approved stop_and_relaunch diagnosis."""
        # Safety rule 1: auto_relaunch must be enabled
        if not self.config.auto_relaunch:
            logger.info("auto_relaunch is disabled; skipping relaunch for %s", diagnosis.diagnosis_id)
            return

        # Safety rule 2: diagnosis must be approved
        if diagnosis.approved is not True:
            logger.info("Diagnosis %s not approved; skipping relaunch", diagnosis.diagnosis_id)
            return

        # Safety rule 3: daily relaunch limit
        daily_count = self.store.get_daily_relaunch_count()
        if daily_count >= self.config.daily_relaunch_limit:
            logger.warning(
                "Daily relaunch limit (%d) reached; skipping", self.config.daily_relaunch_limit
            )
            return

        # Safety rule 4: per-run relaunch limit (max 3 total)
        total_count = self.store.get_total_relaunch_count(snapshot.run_id)
        if total_count >= 3:
            logger.warning(
                "Run %s has been relaunched %d times total; skipping",
                snapshot.run_id, total_count,
            )
            return

        # Safety rule 5: launch_cmd must be present
        launch_cmd = snapshot.config.get("launch_cmd")
        if not launch_cmd:
            logger.error(
                "launch_cmd missing from run config for %s; cannot relaunch. "
                "Add launch_cmd to your W&B run config.",
                snapshot.run_id,
            )
            return

        # Check current run state before acting
        try:
            import wandb  # noqa: PLC0415

            api = wandb.Api()
            run = api.run(f"{snapshot.entity}/{snapshot.project}/{snapshot.run_id}")
            if run.state != "running":
                logger.info(
                    "Run %s is no longer running (state: %s); skipping relaunch",
                    snapshot.run_id, run.state,
                )
                return
            run.stop()
            logger.info("Stopped run %s", snapshot.run_id)
        except Exception as exc:
            logger.error("Failed to stop run %s: %s", snapshot.run_id, exc)
            return

        # Write patched config
        patch_path = self._patch_config(diagnosis, snapshot)

        # Launch with patched config
        cmd = str(launch_cmd).replace("{config}", str(patch_path))
        try:
            proc = subprocess.Popen(cmd, shell=True)
            self.store.save_relaunch(diagnosis.diagnosis_id, snapshot.run_id, proc.pid)
            logger.info("Relaunched run %s with PID %d", snapshot.run_id, proc.pid)
            self._notify_relaunch(diagnosis, snapshot, proc.pid)
        except Exception as exc:
            logger.error("Failed to relaunch run %s: %s", snapshot.run_id, exc)

    def _notify_relaunch(
        self, diagnosis: Diagnosis, snapshot: RunSnapshot, pid: int
    ) -> None:
        webhook = self.config.notify_slack_webhook
        message = (
            f"*[W&B Agent]* :white_check_mark: Run `{snapshot.run_name}` relaunched "
            f"(PID {pid}) with patched config after {diagnosis.failure_mode} diagnosis."
        )
        if not webhook:
            logger.info(message)
            return
        self._slack_post(webhook, message)
