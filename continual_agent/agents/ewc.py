"""
Elastic Weight Consolidation (EWC) for continual learning.

EWC (Kirkpatrick et al., 2017) prevents catastrophic forgetting by adding a
quadratic penalty to the loss that anchors important weights close to their
values after training on a previous task. Importance is measured by the
diagonal of the Fisher information matrix.

Loss = TD_loss + (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2

where:
  F_i      = Fisher information for parameter i (estimated from replay data)
  theta*_i = optimal parameter value after training on the previous task
  lambda   = regularisation strength (higher = less forgetting, slower adaptation)

Reference: Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting
in neural networks. PNAS, 114(13), 3521-3526.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """
    Computes and applies EWC regularisation for a single previous task.

    Usage::

        # After training task k, consolidate:
        ewc = EWC(agent.policy_net, replay_data, lambda_ewc=400)

        # When training task k+1, pass the penalty to the agent update:
        agent.update(ewc_penalty_fn=ewc.penalty)
    """

    def __init__(
        self,
        model: nn.Module,
        replay_data: list[tuple],
        lambda_ewc: float = 400.0,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model:       The policy network after training on the previous task.
            replay_data: List of (state, action, reward, next_state, done) tuples
                         used to estimate the Fisher information matrix.
            lambda_ewc:  Regularisation strength. Higher values preserve previous
                         task performance more strongly.
            device:      Torch device string.
        """
        self.lambda_ewc = lambda_ewc
        self.device = torch.device(device)

        # Store a frozen copy of the optimal parameters theta*
        self._params_star: dict[str, torch.Tensor] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # Estimate diagonal Fisher information matrix
        self._fisher: dict[str, torch.Tensor] = self._compute_fisher(
            model, replay_data
        )

    # ------------------------------------------------------------------
    # Fisher information estimation
    # ------------------------------------------------------------------

    def _compute_fisher(
        self, model: nn.Module, replay_data: list[tuple]
    ) -> dict[str, torch.Tensor]:
        """Estimate the diagonal Fisher information matrix from replay data."""
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        model.eval()
        n_samples = 0

        for state, action, *_ in replay_data:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.LongTensor([action]).to(self.device)

            model.zero_grad()
            logits = model(state_t)

            # Use log-softmax as a proxy for the log-likelihood
            log_probs = F.log_softmax(logits, dim=1)
            log_prob = log_probs[0, action_t[0]]
            log_prob.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()

            n_samples += 1

        if n_samples > 0:
            for n in fisher:
                fisher[n] /= n_samples

        model.train()
        return fisher

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC regularisation penalty for the current model."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self._fisher:
                loss = loss + (
                    self._fisher[n] * (p - self._params_star[n].to(self.device)).pow(2)
                ).sum()
        return (self.lambda_ewc / 2.0) * loss


class MultiTaskEWC:
    """
    Accumulates EWC constraints across multiple previous tasks.

    Each call to consolidate() adds a new EWC object for the just-finished
    task. The combined penalty is the sum of all individual task penalties.
    """

    def __init__(self, lambda_ewc: float = 400.0, device: str = "cpu") -> None:
        self.lambda_ewc = lambda_ewc
        self.device = device
        self._ewc_list: list[EWC] = []

    def consolidate(
        self, model: nn.Module, replay_data: list[tuple]
    ) -> None:
        """Register the current model state as a task to protect."""
        self._ewc_list.append(
            EWC(model, replay_data, self.lambda_ewc, self.device)
        )

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Sum of EWC penalties from all consolidated tasks."""
        if not self._ewc_list:
            return torch.tensor(0.0)
        return sum(ewc.penalty(model) for ewc in self._ewc_list)

    @property
    def n_tasks_consolidated(self) -> int:
        return len(self._ewc_list)
