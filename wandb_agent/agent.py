"""Claude-powered diagnosis agent."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

import re

import anthropic
from openai import OpenAI

from wandb_agent.poller import Diagnosis, RunSnapshot

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert ML training monitor. You receive a snapshot of a live W&B
training run and must diagnose whether it is healthy or has a problem.

Return ONLY a valid JSON object with these exact fields:
{
  "status": "ok" | "warning" | "critical",
  "failure_mode": "none" | "diverging" | "plateau" | "overfitting" | "grad_explosion" | "lr_too_low" | "nan_inf",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-4 sentence plain English explanation>",
  "suggested_action": "none" | "notify" | "patch_config" | "stop_and_relaunch",
  "suggested_diff": {}
}

Diagnosis rubric:
- diverging: training loss increasing monotonically for >20 steps → reduce LR 5-10x, check grad clipping
- nan_inf: loss is NaN or inf → enable grad clipping, check log(0), reduce LR
- plateau: val_loss flat for >50 steps, not near target → LR decay, check data pipeline
- overfitting: train loss falling while val_loss rising for >30 steps → increase regularisation
- grad_explosion: grad_norm >100 and increasing → add/reduce clip_grad_norm threshold
- lr_too_low: loss decreasing very slowly vs early rate → increase LR or use warmup

Action thresholds:
- suggested_action="notify" if confidence >= 0.5
- suggested_action="patch_config" if confidence >= 0.7
- suggested_action="stop_and_relaunch" if confidence >= 0.9

For suggested_diff, only include hyperparameters you recommend changing, e.g.:
{"lr": {"before": 0.001, "after": 0.0002}}

If past diagnoses are provided, take into account whether a previous fix was already tried.
If the run is healthy, return status="ok", failure_mode="none", suggested_action="none".\
"""


def _extract_json(text: str) -> str:
    """Strip markdown code fences and extract the first JSON object found."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _fallback_diagnosis(run_id: str) -> Diagnosis:
    return Diagnosis(
        run_id=run_id,
        diagnosis_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        status="ok",
        failure_mode="none",
        confidence=0.0,
        reasoning="Could not parse Claude response; treating run as healthy to avoid false positives.",
        suggested_action="none",
        suggested_diff={},
        approved=None,
        rejection_reason=None,
    )


class MonitorAgent:
    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        ollama_model: str = "",
        ollama_base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url

    def _call_ollama(self, user_message: str) -> str:
        client = OpenAI(base_url=self.ollama_base_url, api_key="ollama")
        response = client.chat.completions.create(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, user_message: str) -> str:
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    def diagnose(self, snapshot: RunSnapshot, past_diagnoses: list[Diagnosis]) -> Diagnosis:
        """Call the configured LLM to diagnose a run snapshot. Retries once on parse failure."""
        user_message = (
            f"Run name: {snapshot.run_name}\n"
            f"Run ID: {snapshot.run_id}\n"
            f"Config: {json.dumps(snapshot.config, indent=2)}\n\n"
            f"Last {len(snapshot.history)} steps of metrics:\n"
            f"{json.dumps(snapshot.history[-50:], indent=2)}\n\n"
            f"Past diagnoses for this run ({len(past_diagnoses)} total):\n"
            f"{json.dumps([d.model_dump() for d in past_diagnoses[-3:]], indent=2, default=str)}"
        )

        for attempt in range(2):
            try:
                if self.ollama_model:
                    response_text = self._call_ollama(user_message)
                else:
                    response_text = self._call_anthropic(user_message)
                parsed = json.loads(_extract_json(response_text))
                return Diagnosis(
                    run_id=snapshot.run_id,
                    diagnosis_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    status=parsed["status"],
                    failure_mode=parsed["failure_mode"],
                    confidence=float(parsed["confidence"]),
                    reasoning=parsed["reasoning"],
                    suggested_action=parsed["suggested_action"],
                    suggested_diff=parsed.get("suggested_diff", {}),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                if attempt == 0:
                    logger.warning(
                        "Failed to parse diagnosis for run %s (attempt 1): %s — retrying",
                        snapshot.run_id, exc,
                    )
                    continue
                logger.error(
                    "Failed to parse diagnosis for run %s after retry: %s",
                    snapshot.run_id, exc,
                )
                return _fallback_diagnosis(snapshot.run_id)
            except anthropic.APIError as exc:
                logger.error("LLM API error for run %s: %s", snapshot.run_id, exc)
                return _fallback_diagnosis(snapshot.run_id)
            except Exception as exc:
                logger.error("LLM API error for run %s: %s", snapshot.run_id, exc)
                return _fallback_diagnosis(snapshot.run_id)

        return _fallback_diagnosis(snapshot.run_id)
