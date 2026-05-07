# CauAug_DCMIS

The official implementation of our paper
["Causality-Adjusted Data Augmentation for Domain Continual Medical Image Segmentation"](https://ieeexplore.ieee.org/document/11054328),
published in **IEEE Journal of Biomedical and Health Informatics (JBHI), 2025**.

CauAug is the second work in our domain continual medical image segmentation series.
Building upon TED, CauAug studies continual segmentation from a **causal learning
perspective** and focuses on a new question:

> Can both old and new knowledge become biased during continual learning because of
> spurious correlations and domain-specific confounders?

To address this problem, CauAug introduces **causality-adjusted data augmentation**
for jointly optimizing:

- old knowledge retention;
- new knowledge learning;
- continual adaptation under domain shifts.

## 📖 Table of Contents

- [CauAug\_DCMIS](#cauaug_dcmis)
  - [📖 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [What's New](#whats-new)
  - [Research Motivation](#research-motivation)
  - [Method Evolution](#method-evolution)
  - [Key Idea](#key-idea)
  - [Framework](#framework)
  - [Requirements](#requirements)
  - [Project Structure](#project-structure)
  - [Data Preparation](#data-preparation)
  - [Run](#run)
  - [Analysis](#analysis)
  - [Ablation Study](#ablation-study)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Overview

Domain continual medical image segmentation aims to continuously adapt segmentation
models to sequentially arriving medical domains without accessing previous-domain data.

Compared with traditional continual learning scenarios, medical domain shifts are often
caused by:

- different imaging devices;
- different acquisition protocols;
- different hospitals and populations;
- texture and appearance variations.

Although knowledge distillation can alleviate catastrophic forgetting, existing methods
mainly focus on preserving old knowledge and overlook an important issue:

> Both old and new knowledge may simultaneously contain causal bias.

These biases are introduced by spurious correlations and domain-specific confounders,
which may cause the segmentation model to rely on irrelevant local textures or unstable
domain appearance instead of true anatomical structures.

CauAug addresses this problem by introducing causal intervention mechanisms into domain
continual segmentation.

## What's New

This work extends our previous TED framework from a causal perspective.

- **TED_DCMIS** focuses on enhancing old knowledge learning through tri-enhanced
  distillation.
  - 📄 [Paper (MedIA)](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000379)
  - 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/TED_DCMIS)

Our subsequent work further extends CauAug:

- **TKRL_DCMIS** focuses on teacher-originated defects and rectifies inherited
  knowledge gaps and biases inside old teacher models.
  - 📄 [Paper (IEEE JBHI)](https://ieeexplore.ieee.org/document/11359675)
  - 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/TKRL_DCMIS)

## Research Motivation

TED improves old knowledge retention by enhancing:

- diversity;
- transfer accuracy;
- fusion stability.

However, continual segmentation still suffers from a deeper issue.

In domain continual medical image segmentation, the segmentation model may learn
spurious correlations caused by:

- irrelevant local textures;
- domain-specific appearance features.

As a result:

1. **New knowledge learning becomes biased**
   - the model may overfit texture-level confounders;
   - anatomical generalization becomes weak.

2. **Old knowledge retention becomes biased**
   - domain-specific information may dominate knowledge distillation;
   - old knowledge may not be comprehensively preserved.

Therefore, CauAug studies continual segmentation from a causal perspective and attempts
to jointly optimize both old and new knowledge learning under domain shifts.

## Method Evolution

CauAug is the second stage of our research line on domain continual medical image
segmentation.

```text
TED
└── How to better retain old knowledge?
    ├── diversity enhancement
    ├── transfer accuracy enhancement
    └── fusion stability enhancement

CauAug
└── How to causally optimize both old and new knowledge?
    ├── causal intervention
    ├── causal augmentation
    └── confounder disentanglement
```

The overall evolution of this research series is:

```text
TED: Old knowledge retention
  ↓
CauAug: Causal learning of both old and new knowledge
  ↓
TKRL: Rectification of teacher-originated defects
```

Compared with TED, CauAug no longer treats continual learning as only a knowledge
retention problem. Instead, it asks whether the retained knowledge itself may already
contain causal bias caused by domain-specific confounders.

## Key Idea

The core idea of CauAug is:

> Continual segmentation should learn causal anatomical representations instead of
> domain-specific spurious correlations.

To achieve this goal, CauAug introduces causality-adjusted augmentation mechanisms for
both:

- new knowledge generalization;
- old knowledge distillation.

Specifically, CauAug introduces:

- **Texture-Domain Adjustment Hybrid-Scheme (TDAHS)**
  - establishes a domain-continual causal model;
  - identifies confounders in continual segmentation.

- **Cross Kernel Network (CKNet)**
  - removes local texture confounders;
  - improves anatomical generalization for new knowledge learning.

- **Fourier Transformer Generator (FTGen)**
  - selectively restores old domain-specific features;
  - improves comprehensive old knowledge distillation.

## Framework

CauAug introduces a causal intervention framework for domain continual segmentation.

The framework contains two causal pathways:

1. **Anatomy Causality**
   - focuses on anatomical structures;
   - removes irrelevant local texture confounders.

2. **Domain Causality**
   - focuses on domain-specific features;
   - restores representative old-domain knowledge.

The overall optimization objective jointly combines:

- segmentation learning for the current domain;
- causal new knowledge learning;
- causal old knowledge distillation.

<p align="center">
  <img src="figures/framework.png" width="95%">
</p>

## Requirements

- Python 3.8.15
- PyTorch
- CUDA

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
--ablation/
--ablation_results/
--analysis/
--data_prep/
--mp/
--storage/
--README.md
--requirements.txt
--main.py
--get.py
--args.py
--command
```

## Data Preparation

Please refer to the data preparation instructions:

```bash
cat data_prep/readme.md
python data_prep/prostate_prepare.py
python data_prep/cardiacmm_prepare.py
python data_prep/optic_prepare.py
python data_prep/hippocampus_prepare.py
```

## Run

Please check the example commands:

```bash
cat command
```

Example for prostate continual segmentation:

```bash
python main.py --dataset prostate --approach cauaug --epochs 50 \
  --experiment-name prostate-cauaug-unet --backbone unet --device-ids 5 \
  --DomAug --AnaAug
```

## Analysis

```bash
python analysis/eval_dataset.py   # evaluate each dataset and each approach
python analysis/table_figure.py   # generate tables and figures in the paper
python analysis/save_images.py    # save segmentation results
python analysis/effi.py           # generate dynamic continual curves
```

## Ablation Study

```bash
# Ablation study of CKNet for anatomy causality
python ablation/ana_ablation.py

# Ablation study of FTGen for domain causality
python ablation/dom_ablation.py

# Draw continual-learning tables and curves
ablation/abation.ipynb
```

## Acknowledgement

Our code is inspired by
<a href="https://github.com/MECLabTUDA/ACS">ACS</a>
and our previous work
<a href="https://github.com/PerceptionComputingLab/TED_DCMIS">TED</a>.

## Citation

```bibtex
@ARTICLE{11054328,
  author={Zhu, Zhanshi and Dong, Qing and Luo, Gongning and Wang, Wei and Dong, Suyu and Wang, Kuanquan and Tian, Ye and Wang, Guohua and Li, Shuo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Causality-Adjusted Data Augmentation for Domain Continual Medical Image Segmentation}, 
  year={2026},
  volume={30},
  number={2},
  pages={1429-1442},
  keywords={Image segmentation;Biomedical imaging;Anatomical structure;Data augmentation;Data models;Bioinformatics;Adaptation models;Absorption;Electronic mail;Continuing education;Causal data augmentation;domain continual learning;knowledge distillation;medical segmentation},
  doi={10.1109/JBHI.2025.3584068}}

```
