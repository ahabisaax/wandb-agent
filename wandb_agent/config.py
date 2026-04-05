"""AgentConfig pydantic model — separated from cli.py to avoid circular imports."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class ProjectConfig(BaseModel):
    name: str
    poll_interval_s: int = 45
    training_script_path: str = ""
    project_root: str = ""  # root directory to run Claude Code in for fix command


class AgentConfig(BaseModel):
    entity: str
    projects: list[ProjectConfig]
    auto_relaunch: bool = False
    daily_relaunch_limit: int = 3
    notify_slack_webhook: str = ""
    anthropic_api_key: str = ""
    wandb_api_key: str = ""
    approval_server_port: int = 8765
    ollama_model: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    groq_api_key: str = ""
    groq_model: str = ""

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "AgentConfig":
        env_anthropic = os.environ.get("ANTHROPIC_API_KEY")
        if env_anthropic:
            self.anthropic_api_key = env_anthropic
        env_wandb = os.environ.get("WANDB_API_KEY")
        if env_wandb:
            self.wandb_api_key = env_wandb
        env_groq = os.environ.get("GROQ_API_KEY")
        if env_groq:
            self.groq_api_key = env_groq
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
