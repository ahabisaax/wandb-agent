"""Claude-powered diagnosis agent."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import re

import anthropic
from openai import OpenAI

from wandb_agent.poller import Diagnosis, RunSnapshot

logger = logging.getLogger(__name__)

_CONTEXTS_DIR = Path.home() / ".wandb-agent" / "contexts"

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

IMPORTANT rules:
- Base your diagnosis ONLY on observed metric trends in the history. Never flag a hyperparameter
  value as suspicious just because it looks unusual — only diagnose a problem if the metrics
  actually show it happening.
- If the history is short (<20 steps) or metrics look normal, return status="ok".
- failure_mode MUST be exactly one of: none, diverging, plateau, overfitting, grad_explosion,
  lr_too_low, nan_inf. No other values are valid.

Diagnosis rubric (only apply if metrics clearly show the pattern):
- diverging: training loss increasing monotonically for >20 steps
- nan_inf: loss is NaN or inf
- plateau: val_loss flat for >50 steps and not near a reasonable target
- overfitting: train loss falling while val_loss rising consistently for >30 steps
- grad_explosion: grad_norm >100 and increasing over multiple steps
- lr_too_low: loss decreasing extremely slowly compared to the early rate of improvement

Action thresholds:
- suggested_action="notify" if confidence >= 0.5
- suggested_action="patch_config" if confidence >= 0.7
- suggested_action="stop_and_relaunch" if confidence >= 0.9

For suggested_diff, only include hyperparameters you recommend changing, e.g.:
{"lr": {"before": 0.001, "after": 0.0002}}

If past diagnoses are provided, take into account whether a previous fix was already tried.
If the run is healthy, return status="ok", failure_mode="none", suggested_action="none".\
"""

CONTEXT_GENERATION_PROMPT = """\
You are an expert ML engineer. Based on the W&B run configuration, early training metrics,
and optionally the training script below, generate a concise project context document in markdown.

The document should cover:
1. What is being trained (model type, task, dataset if apparent)
2. Key hyperparameters and their current values
3. What metrics are being tracked and what healthy progress looks like for this setup
4. Specific failure modes or risks to watch for given this architecture and optimizer
5. Any notable choices (e.g. scheduler, regularisation, loss function) that affect diagnosis

Be specific and concise. This document will be prepended to an AI monitor's system prompt
to help it give more accurate, project-specific diagnoses.\
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
        reasoning="Could not parse LLM response; treating run as healthy to avoid false positives.",
        suggested_action="none",
        suggested_diff={},
        approved=None,
        rejection_reason=None,
    )


class MonitorAgent:
    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-opus-4-6",
        ollama_model: str = "",
        ollama_base_url: str = "http://localhost:11434/v1",
        groq_api_key: str = "",
        groq_model: str = "",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user_message: str, json_mode: bool = False) -> str:
        """Route to the configured LLM backend."""
        if self.ollama_model:
            return self._call_ollama(system, user_message, json_mode=json_mode)
        elif self.groq_model:
            return self._call_groq(system, user_message, json_mode=json_mode)
        else:
            return self._call_anthropic(system, user_message)

    def _call_ollama(self, system: str, user_message: str, json_mode: bool = False) -> str:
        client = OpenAI(base_url=self.ollama_base_url, api_key="ollama")
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = client.chat.completions.create(
            model=self.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _call_groq(self, system: str, user_message: str, json_mode: bool = False) -> str:
        logger.debug("Groq key in use: %s...%s", self.groq_api_key[:8], self.groq_api_key[-4:])
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.groq_api_key)
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user_message: str) -> str:
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    # ------------------------------------------------------------------
    # Project context
    # ------------------------------------------------------------------

    @staticmethod
    def context_path(project: str) -> Path:
        return _CONTEXTS_DIR / f"{project}.md"

    @staticmethod
    def context_exists(project: str) -> bool:
        return MonitorAgent.context_path(project).exists()

    @staticmethod
    def load_context(project: str) -> str:
        path = MonitorAgent.context_path(project)
        return path.read_text() if path.exists() else ""

    def generate_context(
        self,
        project: str,
        script_content: str = "",
        snapshot: RunSnapshot | None = None,
    ) -> str:
        """Call the LLM to generate a project context document."""
        user_message = f"Project: {project}\n"
        if snapshot:
            user_message += (
                f"Run config: {json.dumps(snapshot.config, indent=2)}\n\n"
                f"Early metrics (first {min(20, len(snapshot.history))} steps):\n"
                f"{json.dumps(snapshot.history[:20], indent=2)}\n"
            )
        if script_content:
            user_message += f"\nTraining script:\n```python\n{script_content}\n```"

        return self._call_llm(CONTEXT_GENERATION_PROMPT, user_message)

    def ensure_context(
        self,
        project: str,
        script_content: str = "",
        snapshot: RunSnapshot | None = None,
    ) -> str:
        """Return the project context, generating and saving it if it doesn't exist yet."""
        if self.context_exists(project):
            return self.load_context(project)

        logger.info("Generating project context for '%s'...", project)
        try:
            context = self.generate_context(project, script_content, snapshot)
            _CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
            self.context_path(project).write_text(context)
            logger.info("Saved project context to %s", self.context_path(project))
            return context
        except Exception as exc:
            logger.warning("Could not generate project context: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def diagnose(
        self,
        snapshot: RunSnapshot,
        past_diagnoses: list[Diagnosis],
        context: str = "",
    ) -> Diagnosis:
        """Call the configured LLM to diagnose a run snapshot. Retries once on parse failure."""
        system = SYSTEM_PROMPT
        if context:
            system = f"## Project Context\n\n{context}\n\n---\n\n{SYSTEM_PROMPT}"

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
                response_text = self._call_llm(system, user_message, json_mode=True)
                logger.debug("LLM raw response: %s", response_text)
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
