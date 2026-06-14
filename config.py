"""Configuration for IBMA: Information Bottleneck-based Multimodal Alignment.

IBMA aligns *two modality views* with the IB upper-bound loss (IBB, see
``losses.py``). The framework itself is modality- and task-agnostic; a concrete
experiment is described by an :class:`IBMAConfig`, which pins down:

  * the label set (and therefore the number of classes),
  * the two modalities being aligned — their raw-input and feature dims,
  * the alignment (IBB) weights and softness.

The **medical CheXpert** setup is provided as one such config
(:func:`chexpert_config`). Adding a new use case is just another factory that
returns an ``IBMAConfig`` — no changes to the engine or loss are needed.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModalitySpec:
    """One modality view in a cross-modal alignment pair.

    Args:
        name: human-readable identifier, e.g. ``"image"`` / ``"text"``.
        raw_dim: flattened dimensionality of the *raw* input, used to build the
            input-space centroids that anchor the IB bound (e.g. ``224*224*3``
            for an RGB image, ``4096`` for an LLM text embedding).
        feature_dim: dimensionality of the encoder output. The two aligned
            modalities must share the same ``feature_dim`` (the IB bound matches
            them cluster-for-cluster).
        is_task_head: whether this modality's encoder also produces the task
            logits (the classification head). Exactly one modality in a pair
            should set this to ``True``.
    """

    name: str
    raw_dim: int
    feature_dim: int
    is_task_head: bool = False


@dataclass
class IBMAConfig:
    """A full IBMA experiment specification (modality-agnostic)."""

    label_names: List[str]
    modality_a: ModalitySpec
    modality_b: ModalitySpec

    # Cross-modal alignment (IBB) hyper-parameters.
    ib_weight_a: float = 10.0  # weight on the view-A | view-B bound
    ib_weight_b: float = 10.0  # weight on the view-B | view-A bound
    ib_beta: float = 1.0       # softness of the cluster-assignment softmax

    def __post_init__(self):
        if self.modality_a.feature_dim != self.modality_b.feature_dim:
            raise ValueError(
                "Aligned modalities must share feature_dim, got "
                f"{self.modality_a.feature_dim} vs {self.modality_b.feature_dim}."
            )

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


# --------------------------------------------------------------------------- #
# Use cases. Each factory returns an IBMAConfig; the medical one is the first. #
# --------------------------------------------------------------------------- #

# CheXpert "competition" labels (multi-label chest X-ray classification).
CHEXPERT_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]


def chexpert_config(input_size: int = 224,
                    text_embed_dim: int = 4096,
                    feature_dim: int = 768,
                    ib_weight_a: float = 10.0,
                    ib_weight_b: float = 10.0,
                    ib_beta: float = 1.0) -> IBMAConfig:
    """Medical use case: chest X-ray image ↔ radiology-report text embedding.

    * view A — image: a ViT (Medical-MAE pretrained), ``feature_dim`` output.
    * view B — text: an NV-Embed report embedding passed through an adapter to
      the same ``feature_dim``.
    """
    return IBMAConfig(
        label_names=list(CHEXPERT_LABELS),
        modality_a=ModalitySpec(
            name="image",
            raw_dim=input_size * input_size * 3,
            feature_dim=feature_dim,
            is_task_head=True,
        ),
        modality_b=ModalitySpec(
            name="text",
            raw_dim=text_embed_dim,
            feature_dim=feature_dim,
        ),
        ib_weight_a=ib_weight_a,
        ib_weight_b=ib_weight_b,
        ib_beta=ib_beta,
    )
