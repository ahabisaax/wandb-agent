# wandb-agent

A lightweight agent that watches your Weights & Biases training runs and uses an LLM to diagnose problems online. When something looks wrong  (e.g. diverging loss, exploding gradients, overfitting) it tells you, and optionally stops and relaunches the run with a patched config.

It runs on your machine alongside your training jobs.

---

## What it does

Every poll cycle (default: 45 seconds), the agent fetches all running W&B runs in your configured projects. For each run it pulls the last 200 steps of training metrics and sends them to an LLM with a structured diagnosis prompt. The model returns a verdict: healthy, warning, or critical, along with a suggested config change if relevant.

Depending on confidence:

- **>= 50% confidence** — sends you a notification (Slack or logs)
- **>= 70% confidence** — writes a patched config YAML to `~/.wandb-agent/patches/`
- **>= 90% confidence** — asks for your approval to stop the run and relaunch it with the fix applied

The relaunch step requires explicit approval and is off by default. You can approve via the CLI or a local HTTP endpoint.

---

## Supported LLM backends

| Backend | How to configure |
|---|---|
| Ollama (local) | Set `ollama_model` in config, run `ollama serve` |
| Anthropic (Claude) | Set `anthropic_api_key` in config or via env var |

Ollama is the easiest way to get started for free. `llama3.2` works well for this task.

---

## Install

```bash
git clone <repo>
cd wandb-agent
pip install -e .
```

---

## Setup

```bash
cp config.example.yaml ~/.wandb-agent/config.yaml
$EDITOR ~/.wandb-agent/config.yaml
```

Minimum fields:

```yaml
entity: your-wandb-entity
projects:
  - name: your-project
    poll_interval_s: 45

wandb_api_key: ""   # or set WANDB_API_KEY env var

# pick one:
ollama_model: "llama3.2"
# anthropic_api_key: ""  # or set ANTHROPIC_API_KEY env var
```

Optional Slack notifications:

```yaml
notify_slack_webhook: "https://hooks.slack.com/services/..."
```

---

## Running

If using Ollama, start it first:

```bash
ollama serve
```

Then start the agent:

```bash
wandb-agent monitor
```

This blocks and polls continuously. Ctrl-C to stop cleanly.

---

## CLI reference

```bash
wandb-agent monitor                        # start the polling loop
wandb-agent status                         # latest diagnosis per tracked run
wandb-agent pending                        # list runs awaiting relaunch approval
wandb-agent approve <diagnosis_id>         # approve a stop-and-relaunch
wandb-agent reject <diagnosis_id>          # reject it, optionally with a reason
wandb-agent reject <diagnosis_id> --reason "False positive"
```

Approvals can also be sent over HTTP while the agent is running:

```bash
curl -X POST http://localhost:8765/approve/<diagnosis_id>
curl -X POST "http://localhost:8765/reject/<diagnosis_id>?reason=False+positive"
curl http://localhost:8765/pending
```

---

## Stop and relaunch

To enable automatic run relaunching, two things are required:

1. Set `auto_relaunch: true` in your config
2. Add a `launch_cmd` to your W&B run config:

```python
wandb.init(config={
    "lr": 0.001,
    "launch_cmd": "python train.py --config {config}"
})
```

`{config}` is substituted with the path to the patched YAML at relaunch time.

When the agent reaches >= 90% confidence on a diagnosis, it will notify you and wait for approval. Once approved, the next poll cycle stops the run and relaunches it with the suggested fix applied.

Safety limits (always enforced, regardless of config):

- `auto_relaunch` must be explicitly set to `true`
- Max 3 relaunches per run total
- Max `daily_relaunch_limit` relaunches across all runs per 24 hours (default: 3)
- `launch_cmd` must be present in the run's W&B config

---

## Config reference

```yaml
entity: your-wandb-entity
projects:
  - name: your-project
    poll_interval_s: 45        # polling frequency in seconds

auto_relaunch: false           # set true only after testing
daily_relaunch_limit: 3        # max relaunches across all runs per 24h
notify_slack_webhook: ""       # Slack incoming webhook URL

anthropic_api_key: ""          # or ANTHROPIC_API_KEY env var
wandb_api_key: ""              # or WANDB_API_KEY env var

ollama_model: ""               # e.g. llama3.2 — takes priority over Anthropic if set
ollama_base_url: "http://localhost:11434/v1"

approval_server_port: 8765
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Architecture

```
wandb_agent/
├── config.py     — AgentConfig pydantic model
├── poller.py     — RunSnapshot + Diagnosis models, WandbPoller
├── agent.py      — MonitorAgent (Ollama or Claude diagnosis)
├── store.py      — SQLite persistence (~/.wandb-agent/store.db)
├── executor.py   — ActionExecutor (notify / patch / relaunch)
├── approval.py   — FastAPI approval server (port 8765)
└── cli.py        — Typer CLI commands
```
