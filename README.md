# Osteosarcoma-Model
This repository contains the code and model resources used for multimodal
radiogenomic analysis. The workflow includes tumor segmentation, radiomics
feature extraction, deep-learning feature extraction, multimodal feature fusion,
genomic pathway analysis, survival-related modeling, and visualization.

The code is organized to support reproducible research and manuscript
code-availability requirements. Clinical images, masks, labels, and genomic
matrices are not included because they may contain sensitive patient-level
information.

## Repository Structure

```text
.
├── Fusion Model/
│   ├── extract_dl_features.py       # Extract 3D deep-learning image features
│   ├── features_fusion.py           # Multimodal transformer fusion model
│   ├── model_train.py               # Stratified K-fold training script
│   ├── pretrained_ckpt .pt          # Pretrained Swin Transformer checkpoint
│   └── swinMM.py                    # Swin Transformer feature encoder
├── Pyradiomics/
│   ├── extract_radiomics_features.py # PyRadiomics feature extraction
│   ├── cox_auc.R                    # Cox model and AUC analysis
│   └── cox_ZsUnilasAIC.R            # Cox feature selection/modeling
├── Genomics Analysis/
│   ├── DEG/Differentially_expr_gene.R
│   ├── GSEA/GSEA.R
│   ├── GSVA/GSVA.R
│   └── ssGSEA/ssGSEA.R
└── Plot/
    ├── gradcam.py                   # Grad-CAM utility
    └── main_file.py                 # Grad-CAM visualization script
```
## Environment

The Python code was developed with PyTorch, MONAI, nnU-Net, PyRadiomics, and
common scientific Python packages. A CUDA-enabled GPU is recommended for
deep-learning feature extraction, model training, and Grad-CAM visualization.

Install the core Python dependencies:

```bash
pip install torch torchvision torchaudio
pip install monai transformers timm scikit-learn pandas numpy scipy tensorboard
pip install SimpleITK pyradiomics nibabel opencv-python
```

The R scripts require R and common Bioconductor/CRAN packages, including:

```r
DESeq2, edgeR, limma, dplyr, ggplot2, pheatmap, RColorBrewer,
VennDiagram, readr, tibble, stringr, scales, ggrepel, readxl,
clusterProfiler, GSVA, survival, survminer, timeROC
```

Most R scripts automatically check and install missing packages when executed.
## Radiomics Feature Extraction

Radiomics features are extracted from paired image and mask volumes using
PyRadiomics. The script resamples each image-mask pair to a fixed matrix size
and exports scalar radiomics features to CSV.

```bash
cd pyradiomics

python extract_radiomics_features.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --output-csv /path/to/radiomics_features.csv \
  --target-size 96 96 96 \
  --label 1 \
  --force-overwrite
```

The output CSV contains one row per patient and a `PatientID` column followed by
numeric radiomics features.

## Deep-Learning Feature Extraction

The script `fusion model/extract_dl_features.py` extracts 768-dimensional
deep-learning features from 3D medical images using the Swin Transformer encoder
defined in `swinMM.py`.

```bash
cd "fusion model"

python extract_dl_features.py \
  --image-dir /path/to/images \
  --roi-dir /path/to/masks \
  --label-xlsx /path/to/labels.xlsx \
  --checkpoint "pretrained_ckpt .pt" \
  --output-json /path/to/dl_features.json \
  --input-mode image \
  --batch-size 1
```
## Multimodal Fusion Model

The multimodal classifier is implemented in `fusion model/features_fusion.py`.
It combines:

1. Global image deep-learning features.
2. Selected radiomics features.
3. Tumor-region deep-learning features.

The model projects radiomics features into the transformer embedding space,
stacks the three modality tokens, refines them with ViT encoder blocks, and
outputs binary classification logits.

Expected fused feature layout:

```text
[image_768, radiomics_7, tumor_768]
```

Therefore, each sample should have a concatenated feature vector of length
1543.
### Differential Expression Analysis

```bash
Rscript "genomics analysis/DEG/Differentially_expr_gene.R"
```

This script supports differential expression analysis using common R packages
`DESeq2`. Users should update the input expression
matrix path and sample grouping path inside the script before running.

### GSEA

```bash
Rscript "genomics analysis/GSEA/GSEA.R"
```

The GSEA folder includes the corresponding GMT file:

```text
c2_c5_h.all.v7.5.1.symbols.gmt
```

### GSVA

```bash
Rscript "genomics analysis/GSVA/GSVA.R"
```

The GSVA folder includes pathway gene-set files:

```text
c2.all.v7.5.1.symbols.gmt
c5.all.v7.5.1.symbols.gmt
c7.all.v7.5.1.symbols.gmt
```

### ssGSEA

```bash
Rscript "genomics analysis/ssGSEA/ssGSEA.R"
```

The ssGSEA folder includes:

```text
28immune cell.gmt
```

## Cox and Time-Dependent AUC Analysis

Cox regression and time-dependent AUC analysis scripts are provided in the
`pyradiomics/` folder:

```bash
Rscript pyradiomics/cox_auc.R
Rscript pyradiomics/cox_ZsUnilasAIC.R
```

Before running, update the file paths inside each R script to point to your
clinical outcome table and feature matrix.
## Grad-CAM Visualization

Grad-CAM utilities are provided in `plot/`.

```bash
cd plot
python main_file.py
```

Before execution, update the input image folder, output folder, pretrained Swin
checkpoint path, and classifier checkpoint path in `main_file.py`.

The script produces slice-wise Grad-CAM heatmap overlays for NIfTI volumes.

## Reproducibility

The Python training and feature extraction scripts set random seeds by default.
For deterministic behavior, the default seed is:

```text
42
```

Note that exact reproducibility can still depend on CUDA, cuDNN, MONAI, PyTorch,
and hardware versions.
## License

Please specify the license before public release. For academic code release, a
common choice is the MIT License or Apache License 2.0. Some components may be
subject to third-party licenses from MONAI, nnU-Net, PyRadiomics, Hugging Face,
and related packages.
## Citation
