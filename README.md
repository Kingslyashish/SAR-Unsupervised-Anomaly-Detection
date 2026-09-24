# Site-Dependent VAE and AAE Performance for Unsupervised SAR Oil-Spill Detection

Research code and supporting materials for the study comparing a Variational Autoencoder (VAE) and an Adversarial Autoencoder (AAE) for unsupervised oil-spill anomaly detection in SAR imagery.

## Included

- `src/unified_sar_experiment.py` — unified five-seed training/evaluation pipeline used for the corrected experiments.
- `data/metadata/` — Wakashio split/label metadata and the DARTIS-2019 metadata table; raw imagery is not redistributed.
- `results/metrics/` — corrected per-seed and aggregate results reported in the manuscript.
- `paper/` — final manuscript PDF/DOCX and corrected figures.
- `scripts/` — reproducibility notes/wrappers.

## Experimental protocol

- Models: SAR-VAE and SAR-AAE with the same convolutional encoder/decoder backbone.
- Latent dimension: 64.
- Input size: 256 × 256 grayscale.
- Seeds: 42, 123, 456, 789, 999.
- VAE: MSE + KL objective, maximum 100 epochs, early stopping patience 10, gradient clipping at norm 5.0.
- AAE: 50 epochs with adversarial latent regularization.
- VAE evaluation uses the latent mean `mu` for deterministic reconstruction scores.
- Reconstruction MSE is computed per patch.
- Validation-set Youden-J threshold selection is performed independently for each model/seed.
- Reconstruction-error polarity is selected on validation data and then fixed for test evaluation.
- Test labels are not used to select threshold or polarity.

## Dataset protocol

### MV Wakashio

- 128 × 128 source patches, 64-pixel stride (50% overlap).
- At least 70% surveyed/valid area required.
- Oil-positive if at least 2% of patch pixels are in the oil mask.
- Contiguous 384-pixel spatial blocks used for splitting.
- Final split: 33 train / 59 validation / 92 test.
- Validation: 24 normal / 35 oil.
- Test: 57 normal / 35 oil.
- 184 usable patches overall: 70 oil / 114 normal.

### DARTIS-2019

- 200 sampled images per category (`ow`, `nw`, `oc`, `nc`), fixed random seed 42.
- 240 normal training images.
- 280 validation images: 80 normal + 200 oil.
- 280 test images: 80 normal + 200 oil.
- No oil images are used for training.

## Corrected aggregate results

| Dataset | Model | Patch IoU (mean ± population SD) | Selected AUC (mean ± population SD) |
|---|---|---:|---:|
| Wakashio | VAE | 0.6491 ± 0.0870 | 0.9475 ± 0.0498 |
| Wakashio | AAE | 0.7169 ± 0.0438 | 0.9693 ± 0.0055 |
| DARTIS-2019 | VAE | 0.6538 ± 0.0710 | 0.6585 ± 0.0652 |
| DARTIS-2019 | AAE | 0.5007 ± 0.1350 | 0.6460 ± 0.0647 |

For DARTIS-2019, the mean raw validation AUCs are 0.3415 for VAE and 0.3989 for AAE; validation polarity selection changes the scoring direction before the threshold is fixed for test evaluation.

## Data availability

The repository intentionally does not redistribute the underlying Sentinel-1/TerraSAR-X/ALOS-2 rasters, Wakashio shapefiles, or the full DARTIS image collection. Obtain the datasets from their original sources and follow the metadata/split files in this repository.

DARTIS-2019 is referenced in the paper through DOI: https://doi.org/10.1594/PANGAEA.980773

## Running the experiment

The source script reads the project root from `SAR_BASE` and writes outputs to `SAR_OUT`.

Example:

```bash
export SAR_BASE=/path/to/your/project_data
export SAR_OUT=/path/to/output
python src/unified_sar_experiment.py
```

The expected data layout is:

```text
project_data/
├── wakashio_labels_split.csv
├── dataset_wakashio/
│   └── patches/
│       └── *.npy
└── dataset_DARTIS_2019/
    └── subset_images/
        └── *.jpg
```

The experiment requires Python with PyTorch, NumPy, Pillow, pandas, and scikit-learn.

## Important reproducibility note

`results/metrics/` contains the corrected results used for the final manuscript. Older experimental outputs should not be mixed with these results.

## Citation

If you use this repository, please cite the associated paper once its publication metadata/DOI is available.
