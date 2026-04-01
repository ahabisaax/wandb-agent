"""Unit tests for WandbPoller — mocks wandb.Api()."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from wandb_agent.poller import RunSnapshot, WandbPoller


def _make_mock_run(
    run_id: str = "run123",
    name: str = "test_run",
    state: str = "running",
    config: dict | None = None,
    history_records: list[dict] | None = None,
) -> MagicMock:
    run = MagicMock()
    run.id = run_id
    run.name = name
    run.state = state
    run.config = config or {"lr": 0.001}
    run.tags = ["experiment"]
    run.system_metrics = {"gpu_util": 90.0}

    history_df = MagicMock()
    history_df.to_dict.return_value = history_records or [
        {"step": 1, "loss": 1.0, "val_loss": 1.1, "grad_norm": 0.9, "lr": 0.001, "epoch": 0}
    ]
    run.history.return_value = history_df

    return run


def test_poll_returns_snapshots():
    mock_run = _make_mock_run()
    mock_api = MagicMock()
    mock_api.runs.return_value = [mock_run]

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="test-entity", projects=["test-project"])
        snapshots = poller.poll()

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.run_id == "run123"
    assert snap.run_name == "test_run"
    assert snap.project == "test-project"
    assert snap.entity == "test-entity"
    assert snap.state == "running"
    assert isinstance(snap.snapshot_at, datetime)


def test_poll_multiple_projects():
    mock_api = MagicMock()
    mock_api.runs.side_effect = [
        [_make_mock_run(run_id="run_a", name="run_a")],
        [_make_mock_run(run_id="run_b", name="run_b")],
    ]

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj_a", "proj_b"])
        snapshots = poller.poll()

    assert len(snapshots) == 2
    ids = {s.run_id for s in snapshots}
    assert ids == {"run_a", "run_b"}


def test_poll_skips_failing_run_gracefully():
    bad_run = MagicMock()
    bad_run.id = "bad_run"
    bad_run.name = "bad"
    bad_run.state = "running"
    bad_run.config = {}
    bad_run.tags = []
    bad_run.system_metrics = {}
    bad_run.history.side_effect = RuntimeError("W&B API error")

    good_run = _make_mock_run(run_id="good_run", name="good")

    mock_api = MagicMock()
    mock_api.runs.return_value = [bad_run, good_run]

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj"])
        snapshots = poller.poll()

    # bad_run is skipped, good_run (which also errors on history) — but good_run
    # has a working history mock, so only bad_run is skipped
    assert any(s.run_id == "good_run" for s in snapshots)


def test_poll_handles_project_api_failure():
    mock_api = MagicMock()
    mock_api.runs.side_effect = RuntimeError("Network error")

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj"])
        snapshots = poller.poll()

    assert snapshots == []


def test_poll_applies_rate_limit_backoff():
    mock_api = MagicMock()
    mock_api.runs.side_effect = RuntimeError("rate limit exceeded (429)")

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj"])
        poller.poll()  # triggers backoff
        assert poller._backoff_until > 0

        # Second poll should be skipped due to backoff
        snapshots = poller.poll()

    assert snapshots == []


def test_snapshot_config_extracted():
    mock_run = _make_mock_run(config={"lr": 0.01, "batch_size": 64, "launch_cmd": "python train.py"})
    mock_api = MagicMock()
    mock_api.runs.return_value = [mock_run]

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj"])
        snapshots = poller.poll()

    assert snapshots[0].config["lr"] == 0.01
    assert snapshots[0].config["launch_cmd"] == "python train.py"


def test_snapshot_slurm_job_id_populated():
    mock_run = _make_mock_run(config={"lr": 0.001, "SLURM_JOB_ID": "12345"})
    mock_api = MagicMock()
    mock_api.runs.return_value = [mock_run]

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj"])
        snapshots = poller.poll()

    assert snapshots[0].slurm_job_id == "12345"


def test_snapshot_history_records_passed():
    records = [
        {"step": i, "loss": 1.0 - i * 0.01, "val_loss": 1.1 - i * 0.01, "grad_norm": 1.0, "lr": 0.001, "epoch": 0}
        for i in range(1, 11)
    ]
    mock_run = _make_mock_run(history_records=records)
    mock_api = MagicMock()
    mock_api.runs.return_value = [mock_run]

    with patch("wandb.Api", return_value=mock_api):
        poller = WandbPoller(entity="entity", projects=["proj"])
        snapshots = poller.poll()

    assert len(snapshots[0].history) == 10
    assert snapshots[0].history[0]["step"] == 1
