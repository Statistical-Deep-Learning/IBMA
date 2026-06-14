"""Reusable variational upper bound on the Information Bottleneck objective.

Generalizes ``get_IB`` / ``get_IB_Loss`` in ``engine.py``:
  * any number of classes (the original hard-codes 5)
  * any number of clusters per class K (the original uses K=2: centroid + anti-centroid)
  * any feature dimensionality, and either one- or two-view setups

Math (per class, per view A conditioned on view B):

	bound = E_{i~p_A, j~p_x}[log p_x(j)]  -  E_{i~p_A, j~p_B}[log q(j | i)]

where p_A, p_B are the soft cluster assignments of the encoded features for
each view, p_x is the soft assignment of the raw input, and q is estimated
empirically from the joint co-occurrence of (p_A, p_B) within the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cluster_scores(features, centroids, beta=1.0, normalize=True):
	if normalize:
		features = F.normalize(features, p=2, dim=1)
	score = -beta * torch.cdist(features, centroids, p=2) ** 2
	return torch.softmax(score, dim=-1)


def estimate_q(scores_first, scores_second, eps=1e-9):
	denom = scores_second.sum(dim=0).clamp_min(eps)
	return (scores_first.transpose(0, 1) @ scores_second) / denom


def ib_upper_bound(input_scores, scores_first, scores_second, q, eps=1e-9):
	log_input = torch.log(input_scores.clamp_min(eps))
	log_q = torch.log(q.clamp_min(eps))
	term_1 = torch.einsum('ni,nj,nj->n', scores_first, input_scores, log_input)
	term_2 = torch.einsum('ni,nj,ij->n', scores_first, scores_second, log_q)
	return (term_1 - term_2).mean()


class IBUpperBoundLoss(nn.Module):
	"""Multi-class variational IB upper bound for one or two views.

	Args:
		beta: softness of the cluster-assignment softmax.
		symmetric: if True, returns ``(loss_a, loss_b)`` — the bound for view A
			conditioned on B and vice versa. If False, only A | B.
		reduction: ``"mean"`` averages over classes (matches the ``/ 5`` in
			engine.py). ``"sum"`` returns the raw class-wise sum.
		eps: numerical floor for ``log``.

	Forward inputs:
		features_a:      (N, D_a)
		features_b:      (N, D_b)        (ignored if symmetric=False and
		                                  ``input_scores_b`` / ``centroids_b`` are None)
		input_scores_a:  (N, C, K)        soft cluster scores of raw input A
		input_scores_b:  (N, C, K)        soft cluster scores of raw input B
		centroids_a:     (C, K, D_a)      per-class cluster centroids for view A
		centroids_b:     (C, K, D_b)      per-class cluster centroids for view B
	"""

	def __init__(self, beta=1.0, symmetric=True, reduction='mean', eps=1e-9):
		super().__init__()
		if reduction not in ('mean', 'sum'):
			raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
		self.beta = beta
		self.symmetric = symmetric
		self.reduction = reduction
		self.eps = eps

	def per_class(self, features_a, features_b,
	              input_scores_a_cls, input_scores_b_cls,
	              centroids_a_cls, centroids_b_cls):
		scores_a = compute_cluster_scores(features_a, centroids_a_cls, beta=self.beta)
		scores_b = compute_cluster_scores(features_b, centroids_b_cls, beta=self.beta)
		q_a = estimate_q(scores_a, scores_b, eps=self.eps)
		loss_a = ib_upper_bound(input_scores_a_cls, scores_a, scores_b, q_a, eps=self.eps)
		if not self.symmetric:
			return loss_a, None
		q_b = estimate_q(scores_b, scores_a, eps=self.eps)
		loss_b = ib_upper_bound(input_scores_b_cls, scores_b, scores_a, q_b, eps=self.eps)
		return loss_a, loss_b

	def forward(self, features_a, features_b,
	            input_scores_a, input_scores_b,
	            centroids_a, centroids_b):
		num_classes = centroids_a.shape[0]
		loss_a = features_a.new_zeros(())
		loss_b = features_a.new_zeros(())
		for cls in range(num_classes):
			la, lb = self.per_class(
				features_a, features_b,
				input_scores_a[:, cls], input_scores_b[:, cls],
				centroids_a[cls], centroids_b[cls],
			)
			loss_a = loss_a + la
			if lb is not None:
				loss_b = loss_b + lb

		if self.reduction == 'mean':
			loss_a = loss_a / num_classes
			loss_b = loss_b / num_classes

		if self.symmetric:
			return loss_a, loss_b
		return loss_a
