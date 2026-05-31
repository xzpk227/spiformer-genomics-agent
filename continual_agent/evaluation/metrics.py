"""
Continual learning evaluation metrics.

Implements standard continual learning evaluation following Lopez-Paz &
Ranzato (2017) and Chaudhry et al. (2018):

  R[i, j]  = accuracy on task j after training on task i  (i >= j)
  b[j]     = accuracy on task j before any training (random policy baseline)

  Average Accuracy (AA):
      AA = (1/T) * sum_j R[T, j]
      Mean accuracy across all tasks after training on the final task.
      Higher is better.

  Backward Transfer (BWT):
      BWT = (1/(T-1)) * sum_{j<T} (R[T, j] - R[j, j])
      How much training on later tasks hurts earlier tasks.
      Negative BWT = forgetting. BWT = 0 = no forgetting.

  Forward Transfer (FWT):
      FWT = (1/(T-1)) * sum_{j>1} (R[j-1, j] - b[j])
      How much training on earlier tasks helps later ones.
      Positive FWT = positive transfer.
"""

from __future__ import annotations

import numpy as np


class ContinualMetrics:
    """
    Accumulates per-task accuracy measurements and computes continual
    learning metrics once training is complete.

    Usage::

        metrics = ContinualMetrics(n_tasks=3)

        # After training task i and evaluating on task j:
        metrics.record(train_task=i, eval_task=j, accuracy=0.82)

        # After all tasks:
        print(metrics.summary())
    """

    def __init__(self, n_tasks: int) -> None:
        self.n_tasks = n_tasks
        # R[i, j] = accuracy on task j after training up to task i
        self._R: np.ndarray = np.full((n_tasks, n_tasks), np.nan)
        # Baseline accuracy before any training
        self._baseline: np.ndarray = np.full(n_tasks, np.nan)

    def record(self, train_task: int, eval_task: int, accuracy: float) -> None:
        """Record accuracy on eval_task after training up to train_task."""
        self._R[train_task, eval_task] = accuracy

    def set_baseline(self, task_id: int, accuracy: float) -> None:
        """Record random/untrained accuracy on task_id (used for FWT)."""
        self._baseline[task_id] = accuracy

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def average_accuracy(self) -> float:
        """Mean accuracy across all tasks after training on the last task."""
        last_row = self._R[self.n_tasks - 1, :]
        valid = last_row[~np.isnan(last_row)]
        return float(np.mean(valid)) if len(valid) > 0 else float("nan")

    def backward_transfer(self) -> float:
        """
        BWT: mean change in task accuracy after subsequent training.
        Negative = forgetting; zero = no forgetting.
        """
        if self.n_tasks < 2:
            return 0.0
        diffs = []
        for j in range(self.n_tasks - 1):
            r_final = self._R[self.n_tasks - 1, j]
            r_diag = self._R[j, j]
            if not np.isnan(r_final) and not np.isnan(r_diag):
                diffs.append(r_final - r_diag)
        return float(np.mean(diffs)) if diffs else float("nan")

    def forward_transfer(self) -> float:
        """
        FWT: mean improvement on future tasks due to prior training.
        Positive = beneficial transfer; zero = no transfer.
        """
        if self.n_tasks < 2:
            return 0.0
        diffs = []
        for j in range(1, self.n_tasks):
            r_prev = self._R[j - 1, j]
            b_j = self._baseline[j]
            if not np.isnan(r_prev) and not np.isnan(b_j):
                diffs.append(r_prev - b_j)
        return float(np.mean(diffs)) if diffs else float("nan")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def accuracy_matrix(self) -> np.ndarray:
        return self._R.copy()

    def summary(self, task_names: list[str] | None = None) -> str:
        names = task_names or [f"Task {i}" for i in range(self.n_tasks)]
        lines = []

        lines.append("\n" + "=" * 60)
        lines.append("  CONTINUAL LEARNING EVALUATION RESULTS")
        lines.append("=" * 60)

        # Accuracy matrix
        header = f"{'':20s}" + "".join(f"{n:>18s}" for n in names)
        lines.append("\nAccuracy Matrix R[train_task, eval_task]:")
        lines.append(header)
        for i in range(self.n_tasks):
            row = f"After {names[i]:14s}"
            for j in range(self.n_tasks):
                val = self._R[i, j]
                cell = f"{val:.3f}" if not np.isnan(val) else "  -  "
                row += f"{cell:>18s}"
            lines.append(row)

        lines.append("")
        aa = self.average_accuracy()
        bwt = self.backward_transfer()
        fwt = self.forward_transfer()

        lines.append(f"  Average Accuracy (AA)  : {aa:.4f}")
        lines.append(
            f"  Backward Transfer (BWT): {bwt:+.4f}"
            + ("  [forgetting]" if bwt < -0.02 else "  [minimal forgetting]")
        )
        lines.append(
            f"  Forward Transfer  (FWT): {fwt:+.4f}"
            + ("  [positive transfer]" if fwt > 0.01 else "")
        )
        lines.append("=" * 60)
        return "\n".join(lines)


def evaluate_policy(agent, env, n_episodes: int = 50) -> float:
    """
    Evaluate agent accuracy: fraction of episodes where the agent uses
    at least min_required relevant tools before calling 'done'.

    Args:
        agent:      DQNAgent instance (uses greedy_action).
        env:        GenomicsToolEnv instance set to the desired task.
        n_episodes: Number of evaluation episodes.

    Returns:
        Fraction of successful episodes in [0, 1].
    """
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.greedy_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if terminated and info["relevant_used"] >= info["min_required"]:
                successes += 1
    return successes / n_episodes
