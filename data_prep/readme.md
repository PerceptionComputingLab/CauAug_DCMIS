This repository provides data preparation scripts and instructions for four specialized segmentation tasks across multiple domains:

- **Prostate MRI Segmentation** (6 domains)  
- **Hippocampus MRI Segmentation** (3 domains)  
- **Cardiac MRI Segmentation** (4 domains)  
- **Fundus Segmentation** (4 domains)

Each dataset is composed of images from multiple sources (domains), employing distinct imaging protocols and acquisition equipment. All datasets are ultimately cropped and/or resized to **192 × 192** and **intensity-normalized to [0, 1]** before training.

---

## Table of Contents
1. [Prostate MRI Segmentation](#prostate-mri-segmentation)
2. [Hippocampus MRI Segmentation](#hippocampus-mri-segmentation)
3. [Cardiac MRI Segmentation](#cardiac-mri-segmentation)
4. [Fundus Segmentation](#fundus-segmentation)


---

## Prostate MRI Segmentation

**Number of domains:** 6  
**Dataset link**: [Prostate dataset](https://drive.google.com/file/d/1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-/view)

### Preparation Details
1. **Intensity normalization**: Each image is rescaled to [0, 1].  
2. **Renaming**: In the BMC dataset, the folder originally named `Seg` must be renamed to `seg`.  
3. **Relabeling**: For the `RUNMC` and `BMC` datasets, convert the labels to a **binary** segmentation (foreground vs. background).  
4. **Resize**: All images are resized to **192 × 192** pixels in the axial plane.  

### Usage
Run the following command to perform preprocessing:
```bash
python prostate_prepare.py
```

### Notes
- The six domains often referenced are RUNMC, BMC, I2CVB, UCL, BIDMC, and HK.  
- Subject counts in common references: 30 (RUNMC), 30 (BMC), 19 (I2CVB), 13 (UCL), 12 (BIDMC), and 12 (HK).

---

## Hippocampus MRI Segmentation

**Number of domains:** 3  
**Datasets/Links**:  
- [HarP (1)](http://www.hippocampal-protocol.net/SOPs/labels.php#final)  
- [HarP (2)](http://www.hippocampal-protocol.net/SOPs/index.php)  
- [DecathlonHippocampus](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)  
- [DryadHippocampus](https://datadryad.org/stash/dataset/doi:10.5061/dryad.gc72v)

### Preparation Details
1. **Intensity normalization**: Each image is rescaled to [0, 1].  
2. **Crop Volume of Interest (VOI)**: From the full brain to the hippocampal region.  
3. **Merge labels**: Some datasets may have multiple label sets for hippocampus sub-regions; these are merged for a unified label set.  
4. **Resize**: Cropped VOIs are resized to **192 × 192** in the axial plane.

### Usage
Run the following command to perform preprocessing:
```bash
python hippocampus_prepare.py
```

### Notes
- Common references include 195 subjects in Decathlon, 25 in Dryad, and 68 in HarP.  
- Ensure that any additional label merges or re-labeling steps follow the instructions in `hippocampus_prepare.py`.

---

## Cardiac MRI Segmentation

**Task**: Segmentation of the left ventricular endocardium (LV-endo), left ventricular epicardium (LV-epi), and the right ventricular endocardium (RV)  
**Number of domains:** 4  
**Dataset/Link**: [M&Ms (Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image) Dataset](https://www.ub.edu/mnms/)

### Preparation Details
1. **Intensity normalization**: Each image is rescaled to [0, 1].  
2. **Grouping**: The M&Ms dataset includes MRI scans from four vendors (`Siemens`, `Philips`, `GE`, and `Canon`), which constitute the four domains.  
3. **Resize**: All images are resized to **192 × 192** in the axial plane.

### Usage
Run the following command to perform preprocessing:
```bash
python cardiacmm_prepare.py
```

### Notes
- Subject counts are approximately: 95 (Siemens), 125 (Philips), 50 (GE), and 50 (Canon).  
- Make sure domain grouping is correctly applied based on `VendorName` metadata.

---

## Fundus Segmentation

**Task**: Segmentation of the optic disc and optic cup  
**Number of domains:** 4  
**Dataset/Link**: [Fundus dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view?usp=sharing)

### Preparation Details
1. **Intensity normalization**: Each image is rescaled to [0, 1].  
2. **Center-crop**: Images are center-cropped to **800 × 800** around the optic disc.  
3. **Resize**: After cropping, images are resized to **192 × 192**.  
4. **Switch mask to binary**: Ensure the segmentation masks are converted to binary labels (foreground vs. background).

### Usage
Run the following command to perform preprocessing:
```bash
python optic_prepare.py
```

### Notes
- Domain breakdown:
  - **Drishti**: 101 samples  
  - **RIM-ONE**: 159 samples  
  - **Zeiss**: 400 samples  
  - **Canon**: 400 samples  
- Make sure cropping is centered on the optic disc region.