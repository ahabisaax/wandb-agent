"""FastAPI approval webhook server."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Query

from wandb_agent.store import RunStore

logger = logging.getLogger(__name__)

app = FastAPI(title="W&B Agent Approval Server", version="0.1.0")

_store: RunStore | None = None


def init_app(store: RunStore) -> None:
    """Bind a RunStore instance to this FastAPI app."""
    global _store
    _store = store


def _get_store() -> RunStore:
    if _store is None:
        raise RuntimeError("Store not initialized — call init_app() first.")
    return _store


@app.post("/approve/{diagnosis_id}")
def approve(diagnosis_id: str) -> dict:
    """Approve a pending stop_and_relaunch diagnosis."""
    store = _get_store()
    store.update_approval(diagnosis_id, approved=True, reason=None)
    logger.info("Approved diagnosis %s", diagnosis_id)
    return {"status": "approved", "diagnosis_id": diagnosis_id}


@app.post("/reject/{diagnosis_id}")
def reject(
    diagnosis_id: str,
    reason: Optional[str] = Query(None, description="Rejection reason"),
) -> dict:
    """Reject a pending stop_and_relaunch diagnosis."""
    store = _get_store()
    store.update_approval(diagnosis_id, approved=False, reason=reason)
    logger.info("Rejected diagnosis %s (reason: %s)", diagnosis_id, reason)
    return {"status": "rejected", "diagnosis_id": diagnosis_id, "reason": reason}


@app.get("/pending")
def pending() -> list[dict]:
    """List all pending (unapproved) stop_and_relaunch diagnoses."""
    store = _get_store()
    diagnoses = store.get_pending_diagnoses()
    return [d.model_dump(mode="json") for d in diagnoses]
