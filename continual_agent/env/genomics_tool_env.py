"""
Custom Gymnasium environment for genomics tool selection.

The agent learns to select the right sequence of tools from the existing
genomics agent toolkit to complete different research tasks. Three task
types are defined, each requiring a different subset of tools.

This environment serves as the substrate for continual learning experiments:
the agent trains on tasks sequentially and must retain knowledge of earlier
tasks while acquiring new skills.

Tasks
-----
  0  Variant Analysis    : splicing prediction + gene database queries
  1  Literature Search   : literature retrieval + GWAS catalog
  2  Pathway Enrichment  : enrichment analysis + GRN + literature

State (14-dim float32 vector)
-----
  [0:3]   task one-hot encoding
  [3:13]  binary flags for each tool (1 = already used this episode)
  [13]    normalised step count in [0, 1]

Actions (discrete, 11)
-------
  0-9  select tool i
  10   declare task complete ("done")

Rewards
-------
  +1.0  first use of a relevant tool
  -0.3  use of an irrelevant tool
  -0.5  repeat use of an already-used tool
  +2.0  "done" with >= min_tools_required relevant tools used
  -1.0  "done" too early (below threshold)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


TOOL_NAMES = [
    "search_literature",          # 0
    "query_ensembl",              # 1
    "query_ncbi_gene",            # 2
    "query_gwas_catalog",         # 3
    "predict_splicing",           # 4
    "predict_splicing_spliformer",# 5
    "spliformer_motif",           # 6
    "run_enrichment_analysis",    # 7
    "generate_grn",               # 8
    "generate_report",            # 9
]

TASK_NAMES = [
    "variant_analysis",
    "literature_search",
    "pathway_enrichment",
]

# Optimal tool sets per task (tool indices)
TASK_OPTIMAL_TOOLS: dict[int, set[int]] = {
    0: {4, 5, 1, 2},   # variant analysis: splicing models + gene DBs
    1: {0, 3},          # literature search: pubmed + GWAS
    2: {7, 8, 0},       # pathway enrichment: GSEApy + GRN + literature
}

# Tools that are actively harmful for each task (creates cross-task interference)
# Learning task 1/2 will push Q-values for these tools negative, causing
# catastrophic forgetting of task 0 in the baseline agent.
TASK_PENALIZED_TOOLS: dict[int, set[int]] = {
    0: set(),
    1: {4, 5, 6},       # splicing tools hurt literature search
    2: {4, 5, 6, 3},    # splicing + GWAS hurt pathway enrichment
}

MIN_TOOLS_REQUIRED = {0: 3, 1: 2, 2: 2}
MAX_STEPS = 15
N_TOOLS = len(TOOL_NAMES)
N_TASKS = len(TASK_NAMES)
TASK_STATE_DIM = N_TASKS + N_TOOLS + 1   # 14 — task ID visible (task-incremental)
BLIND_STATE_DIM = N_TOOLS + 1            # 11 — task ID hidden  (class-incremental)
N_ACTIONS = N_TOOLS + 1                  # 11 (10 tools + done)


class GenomicsToolEnv(gym.Env):
    """
    Genomics tool selection environment for continual learning research.

    Two observation modes:
      include_task_id=True  (task-incremental)  : task one-hot in state — easy,
                                                   no forgetting expected.
      include_task_id=False (class-incremental) : task hidden from agent — hard,
                                                   genuine weight interference
                                                   and catastrophic forgetting
                                                   without EWC.
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: int = 0, include_task_id: bool = False) -> None:
        super().__init__()
        assert 0 <= task_id < N_TASKS, f"task_id must be in [0, {N_TASKS - 1}]"
        self.task_id = task_id
        self.include_task_id = include_task_id
        self.optimal_tools = TASK_OPTIMAL_TOOLS[task_id]
        self.penalized_tools = TASK_PENALIZED_TOOLS[task_id]
        self.min_required = MIN_TOOLS_REQUIRED[task_id]

        state_dim = TASK_STATE_DIM if include_task_id else BLIND_STATE_DIM
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._tools_used: np.ndarray = np.zeros(N_TOOLS, dtype=np.float32)
        self._step: int = 0
        self._relevant_used: int = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._tools_used = np.zeros(N_TOOLS, dtype=np.float32)
        self._step = 0
        self._relevant_used = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        truncated = False
        reward = 0.0

        if action == N_TOOLS:  # "done" action
            terminated = True
            if self._relevant_used >= self.min_required:
                reward = 2.0
            else:
                reward = -1.0
        else:
            if self._tools_used[action] > 0:
                reward = -0.5   # repeat use
            elif action in self.optimal_tools:
                reward = 1.0    # relevant tool
                self._relevant_used += 1
            elif action in self.penalized_tools:
                reward = -1.5   # actively harmful — creates cross-task interference
            else:
                reward = -0.3   # irrelevant tool
            self._tools_used[action] = 1.0

        self._step += 1
        if self._step >= MAX_STEPS:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "relevant_used": self._relevant_used,
            "min_required": self.min_required,
        }

    def _get_obs(self) -> np.ndarray:
        step_norm = np.array([self._step / MAX_STEPS], dtype=np.float32)
        if self.include_task_id:
            task_enc = np.zeros(N_TASKS, dtype=np.float32)
            task_enc[self.task_id] = 1.0
            return np.concatenate([task_enc, self._tools_used, step_norm])
        return np.concatenate([self._tools_used, step_norm])

    def set_task(self, task_id: int) -> None:
        assert 0 <= task_id < N_TASKS
        self.task_id = task_id
        self.optimal_tools = TASK_OPTIMAL_TOOLS[task_id]
        self.penalized_tools = TASK_PENALIZED_TOOLS[task_id]
        self.min_required = MIN_TOOLS_REQUIRED[task_id]
        self.reset()

    @property
    def state_dim(self) -> int:
        return TASK_STATE_DIM if self.include_task_id else BLIND_STATE_DIM
