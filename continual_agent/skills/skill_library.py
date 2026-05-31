"""
Skill library for storing and retrieving learned genomics tool-selection policies.

After the agent is trained on a task, its policy network weights are saved as
a named "skill". Skills can be loaded later for evaluation, transfer, or
composition into multi-task agents.

This models the concept of a growing skill library in continual/lifelong
learning research: each task trains a new capability that persists without
overwriting previous knowledge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


class SkillLibrary:
    """
    Persistent store of task-specific policy checkpoints.

    Usage::

        lib = SkillLibrary(save_dir="skills/")

        # After training task k:
        lib.save_skill("variant_analysis", agent.policy_net, metadata={
            "task_id": 0,
            "episodes": 500,
            "final_accuracy": 0.83,
        })

        # Load for evaluation or transfer:
        lib.load_skill("variant_analysis", agent.policy_net)
        skills = lib.list_skills()   # ["variant_analysis", ...]
    """

    def __init__(self, save_dir: str = "skills/") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def save_skill(
        self,
        skill_name: str,
        model: nn.Module,
        metadata: dict | None = None,
    ) -> Path:
        """
        Serialize model weights and optional metadata under skill_name.

        Args:
            skill_name: Unique identifier for this skill.
            model:      The trained policy network to save.
            metadata:   Optional dict of training details (task_id, accuracy, etc.).

        Returns:
            Path to the saved checkpoint file.
        """
        weights_path = self.save_dir / f"{skill_name}.pt"
        torch.save(model.state_dict(), weights_path)

        entry = {
            "weights_path": str(weights_path),
            "metadata": metadata or {},
        }
        self._index[skill_name] = entry
        self._save_index()

        return weights_path

    def load_skill(self, skill_name: str, model: nn.Module) -> nn.Module:
        """
        Load saved weights into model in-place and return it.

        Args:
            skill_name: The skill to load.
            model:      A QNetwork instance with matching architecture.

        Returns:
            The model with loaded weights.

        Raises:
            KeyError: If skill_name is not in the library.
            FileNotFoundError: If the weights file is missing.
        """
        if skill_name not in self._index:
            raise KeyError(
                f"Skill '{skill_name}' not found. Available: {self.list_skills()}"
            )
        weights_path = Path(self._index[skill_name]["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file missing: {weights_path}")

        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        return model

    def list_skills(self) -> list[str]:
        """Return names of all stored skills."""
        return list(self._index.keys())

    def get_metadata(self, skill_name: str) -> dict:
        """Return metadata for a stored skill."""
        return self._index[skill_name]["metadata"]

    def summary(self) -> str:
        """Human-readable summary of stored skills."""
        if not self._index:
            return "Skill library is empty."
        lines = ["Skill Library:"]
        for name, entry in self._index.items():
            meta = entry.get("metadata", {})
            acc = meta.get("final_accuracy", "?")
            ep = meta.get("episodes", "?")
            lines.append(f"  {name:30s}  accuracy={acc}  episodes={ep}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.save_dir / "index.json"

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)
