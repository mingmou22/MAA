# MAA
# Meta-Adversarial Attacks (MAA)

Official implementation of **"Meta-Adversarial Attacks: Exploiting the Shared Vulnerabilities of Vision Models"**.

MAA generates adversarial examples **without any target-specific information** — no queries to the victim model, no surrogate model. Perturbations are synthesized in the HSV color space under the guidance of a target image, allocated over a spatial patch graph via betweenness centrality, and propagated by a Chebyshev spectral graph filter whose per-order weights are **optimized per instance**, jointly with the perturbation.

## Method Overview

Per sample, the attack proceeds in four stages:

1. **HSV decomposition & guidance** — both the source and the target (guidance) image are converted to the perceptually decoupled HSV space; foreground masks are obtained from a frozen, off-the-shelf Mask R-CNN (a general-purpose pretrained model that is never a victim).
2. **Spatial graph construction** — each channel is partitioned into patches by a sliding window; patches form an 8-connected grid graph, and betweenness centrality (Brandes' algorithm) assigns each node a structural importance weight.
3. **Spectral perturbation propagation** — a K-order Chebyshev spectral graph filter diffuses the centrality-weighted perturbation over multi-hop neighborhoods. The per-order filter weights θ are *learnable*: within the T attack iterations, θ is updated by Adam through a one-step lookahead on the channel alignment losses, co-optimized with the perturbation itself. The optimization signal comes exclusively from HSV-space alignment objectives — no victim output, gradient, or parameter is ever accessed.
4. **Channel-wise perturbation generation** — channel-specific alignment losses against the target image: circular (fractal) alignment for H, Gram-matrix style alignment for S (frozen VGG-19 features), and Fourier-spectrum alignment with high-frequency suppression for V. The final perturbation is template-modulated and projected onto the RGB ℓ∞ ball.



Requires Python ≥ 3.9, PyTorch ≥ 2.0, and (recommended) a CUDA-capable GPU. The first run downloads pretrained Mask R-CNN and VGG-19 weights from torchvision automatically.

## Quick Start



Key arguments (defaults follow the paper, Table I):

| Argument | Description |
|---|---|---|
| `--img_size` | input resolution |
| `--patch_size` |sliding-window patch size |
| `--stride` | sliding-window stride |
| `--n_steps` | co-optimization iterations T |
| `--eps` | RGB ℓ∞ budget |
| `--cheb_K` | Chebyshev truncation order K |
| `--learn_theta` | 1: per-instance learnable θ; 0: fixed θ_k = 1/(k+1) |
| `--theta_lr` | Adam learning rate for θ |

Setting `--learn_theta 0` reproduces the fixed-filter variant used in the ablation.

## Notes for Reviewers

- This repository currently contains the **complete method implementation** (all modules required by Algorithm 1) together with a minimal single-pair demo, so that the correctness of the proposed components can be inspected.
- The full evaluation pipeline (ImageNet victim suites, baseline re-implementations, and experiment configuration files) will be released upon acceptance of the paper.
- The optimization of θ uses only the HSV-space alignment losses between the source and guidance images; it involves **no** victim-model information at any stage, consistent with the strict black-box threat model defined in the paper.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{maa,
  title   = {Meta-Adversarial Attacks: Exploiting the Shared Vulnerabilities of Vision Models},
  author  = {Wang, Yuefeng and Lin, Weiguo and Xu, Junfeng and Liu, Xiulong},
  journal = {IEEE Transactions on Multimedia},
  note    = {under review},
  year    = {2026}
}
```

## License

This project is released for academic research purposes only.

<img width="1258" height="360" alt="image" src="https://github.com/user-attachments/assets/896cf087-fc10-491b-92c6-e2bb81111880" />
<img width="912" height="258" alt="image" src="https://github.com/user-attachments/assets/e036db24-f391-4ccd-a54b-cdcf985e4b2f" />

