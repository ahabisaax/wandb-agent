"""Fake training script that simulates a diverging loss."""
import argparse
import math
import time

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--diverge_after", type=int, default=20)
args = parser.parse_args()

run = wandb.init(
    project="wandb-agent-test",
    config={
        "lr": args.lr,
        "steps": args.steps,
        "diverge_after": args.diverge_after,
        "launch_cmd": f"python fake_training.py --lr {{config}}",
    },
)

print(f"Starting fake training (lr={args.lr}, diverges after step {args.diverge_after})")

for step in range(args.steps):
    if step < args.diverge_after:
        # Healthy phase: loss decreasing
        loss = 2.0 * math.exp(-0.05 * step) + 0.05 * (step % 3)
        val_loss = loss + 0.1
        grad_norm = 1.0 + 0.1 * step
    else:
        # Diverging phase: loss exploding
        offset = step - args.diverge_after
        loss = 0.5 + 0.08 * offset + 0.02 * offset ** 1.5
        val_loss = loss + 0.2
        grad_norm = 10.0 + 5.0 * offset

    wandb.log({
        "loss": round(loss, 4),
        "val_loss": round(val_loss, 4),
        "grad_norm": round(grad_norm, 2),
        "lr": args.lr,
        "epoch": step // 10,
    }, step=step)

    print(f"step {step:3d}  loss={loss:.4f}  val_loss={val_loss:.4f}  grad_norm={grad_norm:.2f}")
    time.sleep(1)  # slow enough for the agent to catch it mid-run

wandb.finish()
