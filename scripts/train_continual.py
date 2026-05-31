"""
Train a continual learning agent on sequential genomics tool-selection tasks.

Trains a DQN agent on three tasks in order:
  0  Variant Analysis
  1  Literature Search
  2  Pathway Enrichment

After each task, EWC consolidates the Fisher information to protect
previously learned skills. A baseline run (no EWC) is also executed for
comparison, demonstrating catastrophic forgetting.

Usage:
    python scripts/train_continual.py [--episodes N] [--lambda-ewc L] [--seed S]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from continual_agent.agents.dqn_agent import DQNAgent, DQNConfig
from continual_agent.agents.ewc import MultiTaskEWC
from continual_agent.env.genomics_tool_env import (
    GenomicsToolEnv,
    N_ACTIONS,
    BLIND_STATE_DIM,
    TASK_STATE_DIM,
    TASK_NAMES,
    N_TASKS,
)
from continual_agent.evaluation.metrics import ContinualMetrics, evaluate_policy
from continual_agent.skills.skill_library import SkillLibrary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_task(
    agent: DQNAgent,
    env: GenomicsToolEnv,
    task_id: int,
    n_episodes: int,
    ewc: MultiTaskEWC | None = None,
    verbose: bool = True,
) -> list[float]:
    """Train agent on a single task and return per-episode rewards."""
    env.set_task(task_id)
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push(obs, action, reward, next_obs, done)
            agent.update(ewc_penalty_fn=ewc.penalty if ewc else None)
            obs = next_obs
            total_reward += reward

        episode_rewards.append(total_reward)

        if verbose and (ep + 1) % 100 == 0:
            recent = np.mean(episode_rewards[-50:])
            print(
                f"    Episode {ep+1:4d}/{n_episodes}  "
                f"avg_reward={recent:.2f}  epsilon={agent.epsilon:.3f}"
            )

    return episode_rewards


def run_experiment(
    n_episodes: int,
    lambda_ewc: float,
    seed: int,
    use_ewc: bool = True,
) -> ContinualMetrics:
    set_seed(seed)

    # Class-incremental setting: task ID hidden — forces real weight sharing
    # and demonstrates catastrophic forgetting without EWC.
    state_dim = BLIND_STATE_DIM
    config = DQNConfig(state_dim=state_dim, action_dim=N_ACTIONS, hidden_dim=32)
    agent = DQNAgent(config)
    env = GenomicsToolEnv(task_id=0, include_task_id=False)
    ewc = MultiTaskEWC(lambda_ewc=lambda_ewc) if use_ewc else None
    skill_lib = SkillLibrary(save_dir="skills/") if use_ewc else None
    metrics = ContinualMetrics(n_tasks=N_TASKS)

    # Baseline accuracy before any training (random policy)
    for task_id in range(N_TASKS):
        env.set_task(task_id)
        baseline_acc = evaluate_policy(agent, env, n_episodes=50)
        metrics.set_baseline(task_id, baseline_acc)

    # Sequential training
    for task_id in range(N_TASKS):
        label = "EWC" if use_ewc else "Baseline (no EWC)"
        print(f"\n[{label}] Training Task {task_id}: {TASK_NAMES[task_id]}")

        train_task(agent, env, task_id, n_episodes, ewc=ewc)

        # Evaluate on all tasks seen so far
        for eval_task in range(task_id + 1):
            env.set_task(eval_task)
            acc = evaluate_policy(agent, env, n_episodes=100)
            metrics.record(train_task=task_id, eval_task=eval_task, accuracy=acc)
            print(
                f"  Accuracy on {TASK_NAMES[eval_task]:20s}: {acc:.3f}"
                + (" ← just trained" if eval_task == task_id else "")
            )

        # EWC consolidation: protect current task before moving to next
        if use_ewc and ewc is not None and task_id < N_TASKS - 1:
            replay_sample = agent.buffer.get_recent(500)
            ewc.consolidate(agent.policy_net, replay_sample)
            print(
                f"  EWC consolidated {ewc.n_tasks_consolidated} task(s). "
                f"Parameters protected."
            )

        # Save skill checkpoint
        if skill_lib is not None:
            env.set_task(task_id)
            final_acc = metrics.accuracy_matrix()[task_id, task_id]
            skill_lib.save_skill(
                TASK_NAMES[task_id],
                agent.policy_net,
                metadata={
                    "task_id": task_id,
                    "episodes": n_episodes,
                    "final_accuracy": round(final_acc, 4),
                    "lambda_ewc": lambda_ewc,
                },
            )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lambda-ewc", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Genomics Continual Learning Agent")
    print(f"  Episodes per task : {args.episodes}")
    print(f"  EWC lambda        : {args.lambda_ewc}")
    print(f"  Seed              : {args.seed}")
    print("=" * 60)

    # EWC run
    metrics_ewc = run_experiment(
        n_episodes=args.episodes,
        lambda_ewc=args.lambda_ewc,
        seed=args.seed,
        use_ewc=True,
    )

    # Baseline run (no EWC — shows catastrophic forgetting)
    print("\n" + "-" * 60)
    print("Running baseline (no EWC) to measure catastrophic forgetting...")
    metrics_baseline = run_experiment(
        n_episodes=args.episodes,
        lambda_ewc=0.0,
        seed=args.seed,
        use_ewc=False,
    )

    # Print results
    print(metrics_ewc.summary(task_names=TASK_NAMES))

    print("\n--- Baseline (no EWC) ---")
    print(metrics_baseline.summary(task_names=TASK_NAMES))

    # Forgetting comparison
    ewc_bwt = metrics_ewc.backward_transfer()
    base_bwt = metrics_baseline.backward_transfer()
    print(f"\nForgetting reduction from EWC: {base_bwt - ewc_bwt:+.4f}")
    print(
        f"EWC AA={metrics_ewc.average_accuracy():.4f}  "
        f"vs  Baseline AA={metrics_baseline.average_accuracy():.4f}"
    )

    # Skill library summary
    lib = SkillLibrary(save_dir="skills/")
    print(f"\n{lib.summary()}")


if __name__ == "__main__":
    main()
