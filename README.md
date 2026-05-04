<div align="center">

# Kinetic Mining in Context: Few-Shot Action Synthesis via Text-to-Motion Distillation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2512.11654)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat-square&logo=github)](https://lucazzola.github.io/kinemic-page/)
[![Conference](https://img.shields.io/badge/ICPR-2026-purple?style=flat-square)]()

*Official implementation — ICPR 2026 main conference proceedings*

</div>

---

**KineMIC** adapts a pre-trained Text-to-Motion diffusion model into a specialized Action-to-Motion generator for Human Activity Recognition (HAR), using as few as **10 real samples per class**. It leverages CLIP semantic correspondences to mine kinematically relevant motion from a large source dataset, guiding fine-tuning of the generalist backbone via contrastive distillation and LoRA adaptation.

> For method details, results, and animated examples → [project page](https://lucazzola.github.io/kinemic-page/) · [paper](https://arxiv.org/abs/2512.11654)

<br>

## Repository Structure

```
KineMIC/
├── external/
│   ├── motion-diffusion-model/   # adapted MDM — training & sampling scripts
│   └── pyskl/                    # ST-GCN evaluator for downstream HAR
├── scripts/                      # data preprocessing & few-shot split tools
├── data/                         # datasets (NTU60, NTU120, HumanML3D, ...)
├── prep/                         # setup shell scripts
└── docs/
    ├── setup.md                  # environment & data setup
    ├── train.md                  # training reference
    └── sample.md                 # sampling reference
```

<br>

## Getting Started

| | |
|---|---|
| **Setup** | [docs/setup.md](docs/setup.md) — environment, dependencies, data download |
| **Training** | [docs/train.md](docs/train.md) — MDM baseline, KineMIC, ST-GCN evaluator |
| **Sampling** | [docs/sample.md](docs/sample.md) — motion synthesis, synthetic dataset generation |
| **Other** | [docs/other.md](docs/other.md) — few-shot split tools, ST-GCN evaluator, misc utilities |

| MDM | KineMIC (Ours) |
|---|---|
|![MDM example](media/mdm_gen_example.gif)|![Ours example](media/kinemic_gen_example.gif)|

<br>

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{cazzola2026kinemic,
  title     = {Kinetic Mining in Context: Few-Shot Action Synthesis via Text-to-Motion Distillation},
  author    = {Cazzola, Luca and Alboody, Ahed},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```
