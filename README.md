# IBMA: Information Bottleneck-Based Multimodal Alignment

Multimodal learning aims to integrate information from heterogeneous data sources to improve representation quality and downstream task performance. A key challenge lies in aligning modality-specific representations while suppressing modality-dependent noise and redundancy. The Information Bottleneck (IB) principle provides a principled framework for learning task-relevant representations. Existing multimodal IB methods primarily apply the IB principle to fused multimodal representation and rely on restrictive distributional assumptions, such as Gaussian latent priors induced by variational autoencoders, which may not hold in practice.
In this paper, we propose Information Bottleneck–based Multimodal Alignment (IBMA), a novel multimodal learning framework that enforces the IB principle for both the fused multimodal representation and modality-specific representations. IBMA introduces modality-specific representation alignment that guides each modality-specific encoder to learn informative and task-relevant representations aligned with the complementary modality, thereby enhancing cross-modal semantic consistency. Moreover, we derive a novel, efficient, and distribution-free variational upper bound for the IB loss that avoids unrealistic assumptions on latent feature distributions and is readily optimized using standard stochastic gradient descent. Extensive experiments demonstrate that IBMA achieves superior performance compared to existing multimodal learning methods, validating the effectiveness of modality-specific representation alignment.


## Demo run (CheXpert)

```bash
torchrun --nproc_per_node=2 main_finetune.py \
  --train_file chexpert_report.json --test_file chexpert_report_test.json \
  --train_embedding_file passage_embeddings_1.pt --data_path /path/to/CheXpert-v1.0-small \
  --finetune /path/to/vit-b_CXR_0.5M_mae.pth \
  --finetune_adapter run_.../best_auc.pt \
  --image_ib_weight 0.5 --text_ib_weight 0.5
```

