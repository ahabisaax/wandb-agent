"""W&B API polling and core data models."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RunSnapshot(BaseModel):
    run_id: str
    run_name: str
    project: str
    entity: str
    state: str                  # running | finished | crashed | failed
    config: dict                # full hyperparameter dict from W&B
    history: list[dict]         # last 200 steps: [{step, loss, val_loss, grad_norm, lr, ...}]
    system_metrics: dict        # gpu_util, gpu_memory_used, etc.
    tags: list[str]
    slurm_job_id: str | None    # populated if SLURM_JOB_ID in run config (future use)
    snapshot_at: datetime


class Diagnosis(BaseModel):
    run_id: str
    diagnosis_id: str           # uuid4
    timestamp: datetime
    status: Literal["ok", "warning", "critical"]
    failure_mode: Literal[
        "none", "diverging", "plateau", "overfitting",
        "grad_explosion", "lr_too_low", "nan_inf"
    ]
    confidence: float           # 0.0–1.0
    reasoning: str              # 2–4 sentence plain-English explanation
    suggested_action: Literal["none", "notify", "patch_config", "stop_and_relaunch"]
    suggested_diff: dict        # {"lr": {"before": 1e-3, "after": 1e-4}, ...}
    approved: bool | None = None        # None = pending, True = approved, False = rejected
    rejection_reason: str | None = None


class WandbPoller:
    def __init__(self, entity: str, projects: list[str], poll_interval_s: int = 45) -> None:
        self.entity = entity
        self.projects = projects
        self.poll_interval_s = poll_interval_s
        self._backoff_until: float = 0.0
        self._rate_limit_count: int = 0

    def poll(self) -> list[RunSnapshot]:
        """Fetch all running runs across configured projects."""
        # Import inside method to avoid side effects at module level
        import wandb  # noqa: PLC0415

        now = time.time()
        if now < self._backoff_until:
            remaining = int(self._backoff_until - now)
            logger.warning("Rate limit backoff active, skipping poll (%ds remaining)", remaining)
            return []

        api = wandb.Api()
        snapshots: list[RunSnapshot] = []

        for project in self.projects:
            try:
                runs = api.runs(
                    f"{self.entity}/{project}",
                    filters={"state": "running"},
                )
                run_list = list(runs)
                logger.info("Found %d running run(s) in %s/%s", len(run_list), self.entity, project)
                for run in run_list:
                    try:
                        snapshot = self._snapshot_run(run, project)
                        snapshots.append(snapshot)
                        self._rate_limit_count = 0  # reset on success
                    except Exception:
                        logger.warning("Failed to snapshot run %s in %s", run.id, project, exc_info=True)
            except Exception as exc:
                msg = str(exc).lower()
                if "rate limit" in msg or "429" in msg or "too many requests" in msg:
                    backoff = min(300, 30 * (2 ** self._rate_limit_count))
                    self._backoff_until = time.time() + backoff
                    self._rate_limit_count += 1
                    logger.warning(
                        "W&B API rate limited for project %s; backing off %ds", project, backoff
                    )
                else:
                    logger.warning("Failed to fetch runs for project %s: %s", project, exc)

        return snapshots

    def _snapshot_run(self, run: object, project: str) -> RunSnapshot:
        try:
            history_df = run.history(  # type: ignore[attr-defined]
                samples=200,
                keys=["loss", "val_loss", "grad_norm", "lr", "epoch"],
            )
            history_records: list[dict] = (
                history_df.to_dict(orient="records")
                if hasattr(history_df, "to_dict")
                else []
            )
        except Exception:
            logger.debug("Could not fetch history for run %s", run.id, exc_info=True)  # type: ignore[attr-defined]
            history_records = []

        try:
            system_metrics: dict = dict(run.system_metrics or {})  # type: ignore[attr-defined]
        except Exception:
            system_metrics = {}

        config: dict = dict(run.config or {})  # type: ignore[attr-defined]
        slurm_job_id = config.get("SLURM_JOB_ID")

        return RunSnapshot(
            run_id=run.id,  # type: ignore[attr-defined]
            run_name=run.name or run.id,  # type: ignore[attr-defined]
            project=project,
            entity=self.entity,
            state=run.state,  # type: ignore[attr-defined]
            config=config,
            history=history_records,
            system_metrics=system_metrics,
            tags=list(run.tags) if run.tags else [],  # type: ignore[attr-defined]
            slurm_job_id=str(slurm_job_id) if slurm_job_id else None,
            snapshot_at=datetime.utcnow(),
        )
