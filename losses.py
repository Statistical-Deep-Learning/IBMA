"""Cross-modal alignment losses for IBMA.

The canonical alignment objective in IBMA is the **Information Bottleneck upper
bound** — referred to throughout as **IBB** (a.k.a. the "IB upper bound loss").
It is a variational upper bound on the IB objective, applied symmetrically
across two modality *views*: each modality's encoded representation is aligned
to the other through their soft cluster assignments. See ``ib_losses.py`` for
the math and ``IBUpperBoundLoss`` itself.

We deliberately use IBB here rather than a CLUB mutual-information estimator
(``ib_loss.py``): IBB is parameter-free (no auxiliary network to co-train), is a
genuine *bound* on the IB objective, and generalizes cleanly to any number of
classes and any feature dimensionality — which is what makes the framework
modality- and task-agnostic.

``CrossModalIB`` is the thin, training-loop-facing wrapper: it takes the
per-class centroid / anti-centroid tensors that the engine already maintains
(the K=2 cluster setup used throughout IBMA) and hands them to ``IBB`` without
the caller having to reshape anything.
"""

import torch
import torch.nn as nn

from ib_losses import IBUpperBoundLoss

# Aliases used across the project: "IBB" / "IB upper bound loss".
IBB = IBUpperBoundLoss


class CrossModalIB(nn.Module):
    """Pairwise cross-modal alignment via the IB upper bound (IBB).

    Wraps :class:`~ib_losses.IBUpperBoundLoss` so the training engine can pass
    centroids and anti-centroids directly (the two-cluster, ``K=2`` setup used
    throughout IBMA) instead of reshaping into ``(C, K, D)`` by hand. Works for
    any number of classes ``C`` and any feature dimensionality ``D``.

    The IB math is run in float32 regardless of the autocast dtype of the
    incoming features: the bound involves ``log`` / ``softmax`` terms that are
    numerically happier in full precision, and it sidesteps mixed-dtype
    ``einsum`` errors when features arrive in bf16/fp16.

    Forward returns ``(loss_a, loss_b)``: the bound for view A conditioned on B
    and vice-versa (e.g. image-given-text and text-given-image).
    """

    def __init__(self, beta=1.0, reduction='mean', eps=1e-9):
        super().__init__()
        self.ib = IBUpperBoundLoss(
            beta=beta, symmetric=True, reduction=reduction, eps=eps,
        )

    @staticmethod
    def _stack(centroids, anti_centroids):
        # (C, D) + (C, D) -> (C, 2, D); cluster 0 = centroid, 1 = anti-centroid.
        return torch.stack((centroids, anti_centroids), dim=1).float()

    def forward(self, features_a, features_b,
                input_scores_a, input_scores_b,
                centroids_a, anti_centroids_a,
                centroids_b, anti_centroids_b):
        centroids_a = self._stack(centroids_a, anti_centroids_a)
        centroids_b = self._stack(centroids_b, anti_centroids_b)
        return self.ib(
            features_a.float(), features_b.float(),
            input_scores_a.float(), input_scores_b.float(),
            centroids_a, centroids_b,
        )
