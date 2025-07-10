# CauAug_DCMIS
The implementation of our paper ["Causality-Adjusted Data Augmentation for Domain Continual Medical Image Segmentation"](https://ieeexplore.ieee.org/document/11054328),  IEEE Journal of Biomedical and Health Informatics (JBHI).

A repository for CauAug: a framework that mitigates knowledge bias in domain continual medical image segmentation through causality-adjusted data augmentation.

## 📖 Table of Contents
- [Overview](#overview)
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Run](#run)
- [Analysis](#analysis)
- [Ablation Study](#ablation-study)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


This work differs from our previous TED framework—[Paper (MedIA)](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000379), [Code (GitHub)](https://github.com/PerceptionComputingLab/TED_DCMIS)—by exploring domain continual learning from a causal perspective, aiming to address knowledge biases through causality-adjusted data augmentation.


🚀 Stay tuned! Our upcoming work will continue to explore domain continual segmentation with a new perspective and further innovations.


## 🔍 Introduction

In domain continual medical image segmentation, distillation-based methods mitigate catastrophic forgetting by continuously reviewing old knowledge. However, these approaches often exhibit biases towards both new and old knowledge simultaneously due to confounding factors, which can undermine segmentation performance. To address these biases, we propose the Causality-Adjusted Data Augmentation (CauAug) framework, introducing a novel causal intervention strategy called the TextureDomain Adjustment Hybrid-Scheme (TDAHS) alongside two causality-targeted data augmentation approaches: the
Cross Kernel Network (CKNet) and the Fourier Transformer Generator (FTGen). (1) TDAHS establishes a domaincontinual causal model that accounts for two types of knowledge biases by identifying irrelevant local textures (L) and domain-specific features (D) as confounders. It introduces a hybrid causal intervention that combines traditional confounder elimination with a proposed replacement approach to better adapt to domain shifts, thereby promoting causal segmentation. (2) CKNet eliminates confounder L to reduce biases in new knowledge absorption. It decreases reliance on local textures in input images, forcing the model to focus on relevant anatomical structures and thus improving generalization. (3) FTGen
causally intervenes on confounder D by selectively replacing it to alleviate biases that impact old knowledge retention. It restores domain-specific features in images, aiding in the comprehensive distillation of old knowledge. Our experiments show that CauAug significantly mitigates catastrophic forgetting and surpasses existing methods in
various medical image segmentation tasks. 

## 🛠️ Requirements
- Python 3.8.15
- pip install -r requirements.txt

## 🗂️ Project Structure
```
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

## 📂 Data Preparation
```
    cat data_prep/readme.md
    python data_prep/prostate_prepare.py
    python data_prep/cardiacmm_prepare.py
    python data_prep/optic_prepare.py
    \python data_prep/hippocampus_prepare.py
```

## ▶️ Run
```
    cat command
    python main.py --dataset prostate --approach cauaug --epochs 50 --experiment-name prostate-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug

```

## 📊 Analysis
```
    python analysis/eval_dataset.py # evaluate the performance of each dataset and each approach
    python analysis/table_figure.py # generate the table and figure in the paper
    python analysis/save_images.py # save the segmentation results
    python analysis/effi.py # the dynamic curve
```

## 🧪 Ablation Study
```
    # ablation study of the CKNet for anatomy causality 
    python ablation/ana_ablation.py
    # ablation study of the FTGen for domain causality
    python ablation/dom_ablation.py

    # draw tables for displaying continual process
    ablation/abation.ipynb
    
```

## 🙏 Acknowledgement

Our code is inspired from <a href="https://github.com/MECLabTUDA/ACS
">ACS</a> and our previous work <a href="https://github.com/PerceptionComputingLab/TED_DCMIS">TED</a>


## 📑 Citation

```bash 

@ARTICLE{11054328,
  author={Zhu, Zhanshi and Dong, Qing and Luo, Gongning and Wang, Wei and Dong, Suyu and Wang, Kuanquan and Tian, Ye and Wang, Guohua and Li, Shuo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Causality-Adjusted Data Augmentation for Domain Continual Medical Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/JBHI.2025.3584068}}
