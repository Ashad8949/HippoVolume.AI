# HippoVolume.AI — Automated Hippocampal Volume Quantification

> End-to-end deep learning pipeline for hippocampal segmentation from brain MRI, integrated into clinical PACS workflows for Alzheimer's disease tracking.

**Udacity AI for Healthcare Nanodegree — Course 3: 3D Medical Imaging**

| Metric | Value |
|--------|-------|
| Mean Dice Score | **0.900** |
| Mean Jaccard Index | **0.820** |
| Inference Time | **< 5 seconds** (CPU) |
| Framework | PyTorch 1.3/1.4 |
| Architecture | Recursive U-Net (DKFZ) |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Section 1: Data Exploration & Curation](#section-1-data-exploration--curation)
- [Section 2: Model Training](#section-2-model-training)
- [Section 3: Clinical Deployment](#section-3-clinical-deployment)
- [Results](#results)
- [Section 3 Result Images](#section-3-result-images)
- [Reports (LaTeX)](#reports-latex)
- [Stand-Out Suggestions Implemented](#stand-out-suggestions-implemented)
- [Tech Stack](#tech-stack)

---

## Project Overview

The **hippocampus** is a brain structure critical for memory formation. Its progressive shrinkage (atrophy) is one of the earliest biomarkers of **Alzheimer's disease**. Manual segmentation by radiologists takes 30-60 minutes per scan. This project builds a complete AI system that does it in under 5 seconds:

1. **Section 1 — Data Exploration & Curation**: Explore and clean the Medical Decathlon Hippocampus dataset (NIFTI format)
2. **Section 2 — Model Training**: Train a Recursive U-Net for 3-class segmentation (background, anterior hippocampus, posterior hippocampus)
3. **Section 3 — Clinical Deployment**: Integrate the trained model into a hospital PACS environment with automated DICOM report generation

![Hippocampus](./readme.img/Hippocampus_small.gif)

---

## Architecture

```
┌──────────────────────────── TRAINING PIPELINE ────────────────────────────┐
│                                                                           │
│  NIFTI Volumes ──► EDA & QC ──► Clean Data ──► DataLoader ──► U-Net      │
│  (394 files)       (Sec. 1)     (260 vols)     (9198 slices)  Training   │
│                                                                  │        │
│                                                             model.pth     │
│                                                                  │        │
└──────────────────────────────────────────────────────────────────┼────────┘
                                                                   │
                                                                   ▼
┌──────────────────────────── CLINICAL PIPELINE ────────────────────────────┐
│                                                                           │
│  MRI Scanner ──► Orthanc PACS ──► Lua Routing ──► storescp Listener      │
│  (storescu)      (port 4242)      (auto-forward)   (port 106)            │
│                       ▲                                   │               │
│                       │                                   ▼               │
│                  OHIF Viewer ◄── storescu ◄── DICOM Report ◄── U-Net     │
│                  (port 3000)     (send back)  (1000×1000)     Inference   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
nd320-c3-3d-imaging-starter/
├── data/
│   ├── TrainingSet/
│   │   ├── images/              # 260 NIFTI hippocampus volumes
│   │   └── labels/              # Corresponding segmentation masks
│   └── TestVolumes/
│       ├── Study1/              # Full DICOM brain MRI studies
│       ├── Study2/
│       └── Study3/
├── section1/                    # Data Exploration
│   ├── Final Project EDA.ipynb
│   └── out/                     # Cleaned, validated volumes
├── section2/                    # Model Training
│   └── src/
│       ├── run_ml_pipeline.py           # Main training entry point
│       ├── data_prep/
│       │   ├── HippocampusDatasetLoader.py  # Load & normalize NIFTI
│       │   └── SlicesDataset.py             # 2D slice PyTorch Dataset
│       ├── experiments/
│       │   └── UNetExperiment.py        # Train/val/test loops
│       ├── networks/
│       │   └── RecursiveUNet.py         # U-Net architecture (DKFZ)
│       ├── inference/
│       │   └── UNetInferenceAgent.py    # 3D volume inference
│       ├── utils/
│       │   ├── utils.py                 # med_reshape, TensorBoard
│       │   └── volume_stats.py          # Dice, Jaccard, Sensitivity
│       └── runs/                        # TensorBoard logs
├── section3/                    # Clinical Deployment
│   └── src/
│       ├── inference_dcm.py             # Main DICOM inference pipeline
│       ├── inference/
│       │   └── UNetInferenceAgent.py    # Inference with padding/cropping
│       ├── networks/
│       │   └── RecursiveUNet.py
│       ├── deploy_scripts/
│       │   ├── route_dicoms.lua         # Orthanc auto-routing
│       │   ├── start_listener.sh        # Start DICOM listener
│       │   ├── send_volume.sh           # Send test study to PACS
│       │   └── send_result.sh           # Send report back to PACS
│       └── assets/
│           └── Roboto-Regular.ttf       # Font for report generation
├── reports/                     # LaTeX Reports
│   ├── comprehensive_guide.tex          # Full technical guide
│   └── academic_report.tex              # Academic paper format
└── README.md                    # This file
```

---

## Prerequisites

- **Python 3.7+**
- **CUDA-capable GPU** (recommended for training; CPU works but is slower)
- **Conda** package manager (recommended)
- For Section 3: **DCMTK** tools (`storescu`, `storescp`), **Orthanc** PACS server

---

## Setup & Installation

### Option 1: Using Conda (Recommended)

**For Section 2 (Training — GPU):**
```bash
cd section2/src
conda env create -f environment.yml
conda activate udacity-section2
```

**For Section 3 (Deployment — CPU):**
```bash
cd section3/src
conda env create -f environment.yml
conda activate udacity-section3
```

### Option 2: Manual Pip Install

```bash
pip install torch torchvision numpy nibabel medpy matplotlib tensorboard pydicom pillow scipy
```

### Install DCMTK (for Section 3)

```bash
# Ubuntu/Debian
sudo apt-get install dcmtk

# macOS
brew install dcmtk
```

---

## Section 1: Data Exploration & Curation

### Purpose
Explore the hippocampus dataset, perform quality checks, visualize volumes, and prepare clean data for training.

### How to Run

```bash
cd section1

# Launch the Jupyter notebook
jupyter notebook "Final Project EDA.ipynb"
```

### What It Does
1. Loads NIFTI volumes using `nibabel`
2. Visualizes axial, coronal, and sagittal slices
3. Computes intensity distributions and volume statistics
4. Identifies missing/corrupted image-label pairs (e.g., `hippocampus_118.nii.gz` has no label)
5. Validates hippocampal volumes against clinical reference range (2,200–4,500 mm³)
6. Exports **260 validated volumes** to `section1/out/`

### Expected Output
```
section1/out/
├── images/    # 260 validated NIFTI volumes
└── labels/    # 260 corresponding segmentation masks
```

---

## Section 2: Model Training

### Purpose
Train a Recursive U-Net for 3-class hippocampal segmentation (background, anterior, posterior).

### How to Run

```bash
cd section2/src

# Standard 3-class training
python run_ml_pipeline.py

# STAND-OUT: Single-class mode (merges anterior + posterior into one class)
python run_ml_pipeline.py --single-class
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Recursive U-Net (DKFZ) |
| Input size | 1 × 64 × 64 |
| Output classes | 3 (background, anterior, posterior) |
| Epochs | 10 |
| Batch size | 8 |
| Learning rate | 0.0002 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau |
| Training time | ~12 minutes (GPU) |

### Monitor Training with TensorBoard

```bash
cd section2/src
tensorboard --logdir runs --bind_all
# Open http://localhost:6006 in browser
```

### Expected Output
```
section2/out/
├── 2026-03-08_0018_Basic_unet/
│   ├── model.pth              # Trained model weights (~120 MB)
│   └── results.json           # Per-volume test metrics
├── training_statistics.txt    # Summary of all training metrics
├── training_report.md         # Efficiency analysis & suggestions
└── clinician_email.md         # Sample clinician communication
```

---

## Section 3: Clinical Deployment

### Purpose
Deploy the trained model into a simulated clinical PACS environment. Automatically process DICOM studies and generate standardized reports.

### Quick Test (Standalone — No PACS Required)

```bash
cd section3/src

# Run inference directly on a test study
python inference_dcm.py ../../data/TestVolumes
```

### Full PACS Deployment (Step-by-Step)

#### Step 1: Start the Orthanc PACS Server

```bash
# Using Docker:
docker run -p 4242:4242 -p 8042:8042 -p 3000:3000 jodogne/orthanc-plugins
```

#### Step 2: Upload the Routing Script

```bash
cd section3/src/deploy_scripts
curl -X POST http://localhost:8042/tools/execute-script \
     --data-binary @route_dicoms.lua -v
```

#### Step 3: Start the AI Listener

```bash
# Edit start_listener.sh to set the output directory, then:
bash start_listener.sh
```

#### Step 4: Send a Test Volume to PACS

```bash
cd section3/src
bash deploy_scripts/send_volume.sh
```

#### Step 5: Run Inference

```bash
cd section3/src
python inference_dcm.py /path/to/dicom/routing/directory
```

#### Step 6: View Results in OHIF

Open `http://localhost:3000` to view the report in the clinical viewer.

### Expected Output

```
section3/out/
├── report.dcm                 # DICOM Secondary Capture report
├── report.png                 # PNG version of the report image
└── segmentation_series/       # DICOM segmentation mask series
    ├── seg_0000.dcm
    ├── seg_0001.dcm
    └── ... (one per slice)
```

---

## Results

### Test Set Performance (52 Volumes)

| Metric | Value |
|--------|-------|
| **Mean Dice Score** | **0.9001** |
| **Mean Jaccard Index** | **0.8195** |
| Mean Sensitivity | ~0.89 |
| Mean Specificity | ~0.998 |
| Best Dice | 0.937 (hippocampus_146) |
| Worst Dice | 0.823 (hippocampus_334) |

### Training Loss Progression

| Epoch | Train Loss | Epoch | Train Loss |
|-------|-----------|-------|-----------|
| 0 | 0.0245 | 5 | 0.0135 |
| 1 | 0.0131 | 6 | 0.0050 |
| 2 | 0.0181 | 7 | 0.0073 |
| 3 | 0.0084 | 8 | 0.0078 |
| 4 | 0.0222 | 9 | 0.0087 |

### Top 5 & Bottom 5 Performing Volumes

| Rank | Volume | Dice | Jaccard |
|------|--------|------|---------|
| 1 | hippocampus_146 | 0.937 | 0.881 |
| 2 | hippocampus_034 | 0.933 | 0.874 |
| 3 | hippocampus_175 | 0.932 | 0.873 |
| 4 | hippocampus_207 | 0.931 | 0.871 |
| 5 | hippocampus_152 | 0.930 | 0.870 |
| ... | ... | ... | ... |
| 48 | hippocampus_070 | 0.858 | 0.752 |
| 49 | hippocampus_042 | 0.832 | 0.712 |
| 50 | hippocampus_125 | 0.827 | 0.705 |
| 51 | hippocampus_305 | 0.826 | 0.703 |
| 52 | hippocampus_334 | 0.823 | 0.699 |

---

## Section 3 Result Images

### DICOM Clinical Report

The system generates a **1000×1000 RGB DICOM report** containing:

- **Patient metadata**: ID, name, study date, modality, institution
- **Volumetric measurements**: Anterior, posterior, and total hippocampal volumes (mm³) with percentage breakdown
- **Clinical assessment**: Color-coded comparison against reference range (2,500–4,500 mm³)
  - 🔴 **Red**: Below normal range (< 2,500 mm³) — consider clinical correlation
  - 🟢 **Green**: Within normal reference range
  - 🟠 **Orange**: Above normal range (> 4,500 mm³) — verify segmentation
- **Axial slice visualizations**: Three representative slices with segmentation overlay
  - **Green** = Anterior hippocampus
  - **Red** = Posterior hippocampus
- **Segmentation quality**: Coverage metric (% of volume segmented)
- **AI disclaimer**: Model info (U-Net, Dice=0.90) and clinical decision support notice

The report is saved as:
- `section3/out/report.dcm` — DICOM Secondary Capture (can be viewed in any DICOM viewer or OHIF)
- `section3/out/report.png` — PNG image for quick viewing

### Segmentation Mask DICOM Series

A separate DICOM series is generated at `section3/out/segmentation_series/` containing:
- One DICOM file per axial slice with RGB color-coded segmentation masks
- Proper `FrameOfReferenceUID` for spatial alignment with original images
- Can be loaded into **3D Slicer** or **Radiant DICOM Viewer** and overlaid on the original MRI using the Fusion feature
- Green = Anterior hippocampus, Red = Posterior hippocampus

---

## Reports (LaTeX)

Two comprehensive LaTeX reports are available in the `reports/` directory:

| Report | File | Description |
|--------|------|-------------|
| **Comprehensive Guide** | `reports/comprehensive_guide.tex` | Full technical guide explaining every component: architecture diagrams, data pipeline, U-Net deep dive, training process, evaluation metrics, clinical deployment, DICOM details, glossary |
| **Academic Paper** | `reports/academic_report.tex` | Academic format with abstract, introduction, methods, results, discussion, conclusion, and references |

### Compile to PDF

```bash
cd reports

# Comprehensive guide (run twice for table of contents)
pdflatex comprehensive_guide.tex
pdflatex comprehensive_guide.tex

# Academic paper (run twice for references)
pdflatex academic_report.tex
pdflatex academic_report.tex
```

Both reports include **TikZ architecture diagrams** of the full pipeline and U-Net architecture.

---

## Stand-Out Suggestions Implemented

- ✅ **Sensitivity and Specificity** metrics for clinical relevance
- ✅ **Per-class Dice** scores (anterior vs posterior hippocampus)
- ✅ **Single-class training mode** via `--single-class` command-line flag
- ✅ **Clinical range assessment** in DICOM report (red/green/orange color-coded)
- ✅ **Segmentation mask DICOM series** for 3D viewer overlay
- ✅ **Multi-criteria series selection** (HippoCrop primary + small-dimension fallback)
- ✅ **Full DICOM PS3.3 compliance** for Secondary Capture IOD (all mandatory modules)
- ✅ **Training efficiency report** with hardware requirements and improvement suggestions
- ✅ **Best/worst volume tracking** for quality analysis

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 1.3 (GPU) / 1.4 (CPU) |
| Architecture | Recursive U-Net (DKFZ) |
| Medical Imaging I/O | nibabel, medpy, pydicom |
| Training Monitoring | TensorBoard |
| PACS Server | Orthanc |
| DICOM Network Tools | DCMTK (storescu / storescp) |
| Clinical Viewer | OHIF |
| Report Generation | Pillow (PIL) |
| Visualization | matplotlib |

---

## Sources

[1] [Medical Segmentation Decathlon — Hippocampus Dataset](http://medicaldecathlon.com/)
[2] [Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015](https://arxiv.org/abs/1505.04597)
[3] [DKFZ Recursive U-Net Implementation](https://github.com/MIC-DKFZ)
[4] [DICOM Standard — Secondary Capture IOD](http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)

## License

This project is part of the Udacity AI for Healthcare Nanodegree. The Recursive U-Net implementation is from DKFZ, licensed under Apache 2.0.
[3] [medicaldecathlon.com/](http://medicaldecathlon.com/)# HippoVolume.AI
