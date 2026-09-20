"""E03: E02 architecture + a within-case listwise ranking term in the loss.

Diagnosis driving this change (see E02 RESULT, "Primary bottleneck"): a constant,
case-independent ranking of the 52 classes by global mean predicted probability scores a
better within-case GT rank than H3 (11.427 vs 11.549) and E01 (11.598 vs 11.866), and is
beaten by E02 by only 0.92 ranks.  52 *independent* unweighted BCE heads at a ~0.6 %
positive rate contain no term that makes classes compete inside a case, so the optimum they
approach is the per-class marginal -- prevalence collapse.

The listwise term is a softmax cross-entropy over the 52 logits of a single case against a
target distribution that is uniform over that case's ground-truth locations.  243 of the 416
release cases carry exactly one location, so for the majority of cases this is precisely
"which location is the aneurysm in".  Cases with no ground truth contribute nothing to it
(there is no correct answer to rank first) and are still fully supervised by BCE.

The two terms are complementary rather than competing: softmax cross-entropy is invariant to
a per-case constant shift of the logits, so it constrains only *relative* within-case
contrast, while BCE constrains the *absolute* level that cross-case (column-wise) decision
policies depend on.  That is why lambda = 1.0 is used unweighted and is not swept.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .topaneu_dilation import TopAneuVesselAwareMeanPoolDilationR1


class TopAneuVesselAwareListwise(TopAneuVesselAwareMeanPoolDilationR1):
    """E02 model with BCE + lambda * within-case listwise ranking loss."""

    def __init__(self, *args, listwise_lambda: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.listwise_lambda = float(listwise_lambda)
        # Populated on every loss() call so the trainer can log the split.
        self.last_loss_parts: dict[str, float] = {}

    @staticmethod
    def _listwise_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Softmax CE over the 52 classes of a case vs a uniform-over-GT target.

        Computed in float32: under fp16 autocast a 52-way log_softmax is close enough to
        the half-precision floor to lose the small logit differences this term exists to
        create.  Cases with no positive label are dropped, not zero-filled, so the mean is
        over supervised cases only.
        """
        positive = targets.sum(dim=1) > 0
        if not bool(positive.any()):
            return logits.sum() * 0.0
        selected_logits = logits[positive].float()
        selected_targets = targets[positive].float()
        q = selected_targets / selected_targets.sum(dim=1, keepdim=True)
        log_p = F.log_softmax(selected_logits, dim=1)
        return -(q * log_p).sum(dim=1).mean()

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != targets.shape or logits.shape[-1] != self.num_outputs:
            raise ValueError("Task 1 targets must be a (B,52) multi-hot tensor")
        target_float = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, target_float, pos_weight=pos_weight)
        listwise = self._listwise_ce(logits, target_float)
        total = bce + self.listwise_lambda * listwise
        self.last_loss_parts = {
            "bce": float(bce.detach()),
            "listwise": float(listwise.detach()),
            "total": float(total.detach()),
        }
        return total
