"""Unit tests for MonitorAgent — mocks Anthropic API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wandb_agent.agent import MonitorAgent
from wandb_agent.poller import Diagnosis, RunSnapshot

_FIXTURES = Path(__file__).parent / "fixtures"


def _load_snapshot(name: str) -> RunSnapshot:
    return RunSnapshot.model_validate_json((_FIXTURES / name).read_text())


def _make_mock_response(payload: dict) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_diagnose_healthy_run():
    snapshot = _load_snapshot("healthy_run.json")
    mock_response = _make_mock_response(
        {
            "status": "ok",
            "failure_mode": "none",
            "confidence": 0.05,
            "reasoning": "Loss is decreasing steadily. The run looks healthy.",
            "suggested_action": "none",
            "suggested_diff": {},
        }
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    assert diagnosis.run_id == "healthy123"
    assert diagnosis.status == "ok"
    assert diagnosis.failure_mode == "none"
    assert diagnosis.suggested_action == "none"
    assert isinstance(diagnosis.diagnosis_id, str) and len(diagnosis.diagnosis_id) > 0


def test_diagnose_diverging_run():
    snapshot = _load_snapshot("diverging_run.json")
    mock_response = _make_mock_response(
        {
            "status": "critical",
            "failure_mode": "diverging",
            "confidence": 0.95,
            "reasoning": (
                "Training loss has increased monotonically for all 25 steps from 0.55 to 1.75. "
                "Gradient norms are also rising, suggesting the learning rate is too high. "
                "Recommend reducing LR by 10x and adding gradient clipping."
            ),
            "suggested_action": "stop_and_relaunch",
            "suggested_diff": {"lr": {"before": 0.01, "after": 0.001}},
        }
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    assert diagnosis.run_id == "diverging456"
    assert diagnosis.status == "critical"
    assert diagnosis.failure_mode == "diverging"
    assert diagnosis.confidence >= 0.9
    assert diagnosis.suggested_action == "stop_and_relaunch"
    assert "lr" in diagnosis.suggested_diff


def test_diagnose_overfitting_run():
    snapshot = _load_snapshot("overfitting_run.json")
    mock_response = _make_mock_response(
        {
            "status": "warning",
            "failure_mode": "overfitting",
            "confidence": 0.85,
            "reasoning": (
                "Training loss has decreased from 1.94 to 0.59, but validation loss increased from 0.83 to 1.80. "
                "This is a clear overfitting pattern over 40 steps. Adding dropout or L2 regularization is recommended."
            ),
            "suggested_action": "patch_config",
            "suggested_diff": {"dropout": {"before": 0.0, "after": 0.3}},
        }
    )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    assert diagnosis.run_id == "overfit789"
    assert diagnosis.status == "warning"
    assert diagnosis.failure_mode == "overfitting"
    assert diagnosis.suggested_action == "patch_config"


# ---------------------------------------------------------------------------
# Parse failure / fallback tests
# ---------------------------------------------------------------------------


def test_diagnose_returns_fallback_on_invalid_json():
    snapshot = _load_snapshot("healthy_run.json")

    bad_block = MagicMock()
    bad_block.type = "text"
    bad_block.text = "This is not JSON at all!"
    mock_response = MagicMock()
    mock_response.content = [bad_block]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    # Fallback: treat as ok with 0.0 confidence
    assert diagnosis.status == "ok"
    assert diagnosis.confidence == 0.0
    assert diagnosis.suggested_action == "none"


def test_diagnose_retries_once_then_falls_back():
    snapshot = _load_snapshot("healthy_run.json")

    bad_block = MagicMock()
    bad_block.type = "text"
    bad_block.text = "Not JSON"
    mock_response = MagicMock()
    mock_response.content = [bad_block]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    # Should have called the API exactly twice (original + retry)
    assert mock_client.messages.create.call_count == 2
    assert diagnosis.confidence == 0.0


def test_diagnose_handles_missing_fields_gracefully():
    snapshot = _load_snapshot("healthy_run.json")

    # Return JSON missing required fields
    incomplete_block = MagicMock()
    incomplete_block.type = "text"
    incomplete_block.text = json.dumps({"status": "ok"})  # missing many fields
    mock_response = MagicMock()
    mock_response.content = [incomplete_block]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    assert diagnosis.status == "ok"
    assert diagnosis.confidence == 0.0  # fallback


def test_diagnose_handles_api_error():
    import anthropic as anthropic_module

    snapshot = _load_snapshot("healthy_run.json")

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic_module.APIConnectionError(
            request=MagicMock()
        )
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, [])

    assert diagnosis.status == "ok"
    assert diagnosis.confidence == 0.0


# ---------------------------------------------------------------------------
# Past diagnoses are passed correctly
# ---------------------------------------------------------------------------


def test_diagnose_includes_past_diagnoses_in_prompt():
    snapshot = _load_snapshot("diverging_run.json")
    mock_response = _make_mock_response(
        {
            "status": "critical",
            "failure_mode": "diverging",
            "confidence": 0.93,
            "reasoning": "Still diverging despite previous LR reduction attempt.",
            "suggested_action": "stop_and_relaunch",
            "suggested_diff": {"lr": {"before": 0.001, "after": 0.0001}},
        }
    )

    past = [
        Diagnosis(
            run_id="diverging456",
            diagnosis_id="prev-id",
            timestamp=snapshot.snapshot_at,
            status="warning",
            failure_mode="diverging",
            confidence=0.7,
            reasoning="LR may be too high.",
            suggested_action="patch_config",
            suggested_diff={"lr": {"before": 0.01, "after": 0.001}},
        )
    ]

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        MockAnthropic.return_value = mock_client

        agent = MonitorAgent(api_key="test-key")
        diagnosis = agent.diagnose(snapshot, past)

    # Verify past diagnoses were included in the user message
    call_kwargs = mock_client.messages.create.call_args
    user_content = call_kwargs[1]["messages"][0]["content"]
    assert "Past diagnoses" in user_content
    assert "prev-id" in user_content

    assert diagnosis.status == "critical"
