# wandb-agent

A lightweight agent that watches your Weights & Biases training runs and uses an LLM to diagnose problems online. When something looks wrong (e.g. diverging loss, exploding gradients, overfitting) it tells you, and optionally stops and relaunches the run with a patched config.

It runs on your machine alongside your training jobs.

---

## What it does

Every poll cycle (default: 45 seconds), the agent fetches all running W&B runs in your configured projects. For each run it pulls the last 200 steps of training metrics and sends them to an LLM with a structured diagnosis prompt. The model returns a verdict: healthy, warning, or critical, along with a suggested config change if relevant.

Depending on confidence:

- **>= 50% confidence** — sends you a notification (Slack or logs)
- **>= 70% confidence** — notifies you and asks for approval to apply a config patch
- **>= 90% confidence** — notifies you and asks for approval to stop the run and relaunch with the fix applied

All config changes and relaunches require explicit user approval. Nothing is applied automatically.

---

## Supported LLM backends

| Backend | How to configure |
|---|---|
| Ollama (local) | Set `ollama_model` in config, run `ollama serve` |
| Groq (free API) | Set `groq_model` and `groq_api_key` in config, or set `GROQ_API_KEY` env var |
| Anthropic (Claude) | Set `anthropic_api_key` in config or set `ANTHROPIC_API_KEY` env var |

Priority: **Ollama > Groq > Anthropic**. The first configured backend wins.

Ollama is the easiest way to get started locally. Groq is the easiest free cloud option — sign up at [console.groq.com](https://console.groq.com) and use `llama-3.3-70b-versatile`.

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

Minimum config:

```yaml
entity: your-wandb-entity
projects:
  - name: your-project
    poll_interval_s: 45

wandb_api_key: ""   # or set WANDB_API_KEY env var

# pick one backend:
ollama_model: "llama3.2"
# groq_model: "llama-3.3-70b-versatile"
# groq_api_key: ""       # or set GROQ_API_KEY env var
# anthropic_api_key: ""  # or set ANTHROPIC_API_KEY env var
```

Optional Slack notifications:

```yaml
notify_slack_webhook: "https://hooks.slack.com/services/..."
```

---

## Connecting to a training repo

Your training script just needs `wandb.log()` calls — the agent never touches your code directly. It only reads from W&B's API.

```python
import wandb
wandb.init(project="my-project")  # must match `name` in config.yaml

# inside your training loop
wandb.log({"loss": loss, "val_loss": val_loss, "lr": lr, "grad_norm": grad_norm})
```

Then start your training and the agent in separate terminals:

```bash
# Terminal 1
python train.py

# Terminal 2
wandb-agent monitor
```

---

## Project context

At startup, the agent automatically generates a context document at `~/.wandb-agent/contexts/<project>.md` for each project that has a `training_script_path` configured. This is produced by the LLM from your training script and gets prepended to the diagnosis system prompt so the model understands your specific setup — architecture, optimizer, expected metric behaviour — rather than reasoning generically.

To enable context generation, set `training_script_path` in your project config:

```yaml
projects:
  - name: my-project
    poll_interval_s: 30
    training_script_path: "~/path/to/train.py"
```

The context is generated once and reused on subsequent runs. To regenerate (e.g. after changing your model or optimizer):

```bash
rm ~/.wandb-agent/contexts/my-project.md
```

It will be regenerated on the next agent startup.

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

This blocks and polls continuously. The startup line tells you which LLM backend is active:

```
Monitoring 1 project(s) for entity 'you'. LLM: groq / llama-3.3-70b-versatile. Approval server on port 8765. Ctrl-C to stop.
```

---

## CLI reference

```bash
wandb-agent monitor                        # start the polling loop
wandb-agent status                         # latest diagnosis per tracked run
wandb-agent pending                        # list runs awaiting approval
wandb-agent approve <diagnosis_id>         # approve a stop-and-relaunch
wandb-agent reject <diagnosis_id>          # reject it, optionally with a reason
wandb-agent reject <diagnosis_id> --reason "False positive"
wandb-agent fix <diagnosis_id>             # invoke Claude Code to fix the diagnosed issue
```

Approvals can also be sent over HTTP while the agent is running:

```bash
curl -X POST http://localhost:8765/approve/<diagnosis_id>
curl -X POST "http://localhost:8765/reject/<diagnosis_id>?reason=False+positive"
curl http://localhost:8765/pending
```

---

## Fixing issues with Claude Code

When the agent diagnoses a problem, you can hand it off directly to Claude Code to propose a fix:

```bash
wandb-agent fix <diagnosis_id>
```

This invokes Claude Code in your project directory with full context:
- The diagnosis (failure mode, reasoning, suggested diff)
- The W&B run config
- The project context document
- The path to your training script

Claude Code then investigates and proposes the appropriate fix — whether that's a hyperparameter change in your experiment config, a code fix in your training script, or something else. You review and approve the diff before anything is applied.

To enable this, set `project_root` in your project config:

```yaml
projects:
  - name: my-project
    project_root: "~/path/to/your/training/repo"
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
- If the run finishes naturally between diagnosis and approval, the relaunch is skipped gracefully

---

## Config reference

```yaml
entity: your-wandb-entity
projects:
  - name: your-project
    poll_interval_s: 45           # polling frequency in seconds
    training_script_path: ""      # optional path to train.py for context generation
    project_root: ""              # project directory for Claude Code fix invocation

auto_relaunch: false              # set true only after testing
daily_relaunch_limit: 3           # max relaunches across all runs per 24h
notify_slack_webhook: ""          # Slack incoming webhook URL

anthropic_api_key: ""             # or ANTHROPIC_API_KEY env var
groq_api_key: ""                  # or GROQ_API_KEY env var
groq_model: ""                    # e.g. llama-3.3-70b-versatile
wandb_api_key: ""                 # or WANDB_API_KEY env var

ollama_model: ""                  # e.g. llama3.2 — takes priority over all if set
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
├── agent.py      — MonitorAgent (LLM diagnosis + project context generation)
├── store.py      — SQLite persistence (~/.wandb-agent/store.db)
├── executor.py   — ActionExecutor (notify / patch / relaunch)
├── approval.py   — FastAPI approval server (port 8765)
└── cli.py        — Typer CLI commands

~/.wandb-agent/
├── config.yaml         — your configuration
├── store.db            — SQLite history of snapshots and diagnoses
├── contexts/           — auto-generated per-project context documents
│   └── <project>.md
└── patches/            — suggested config diffs written on patch_config action
```
