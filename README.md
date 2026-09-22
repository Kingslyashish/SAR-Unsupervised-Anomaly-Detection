# Unsupervised SAR Anomaly Detection for Marine Oil Spills
Official codebase for **"Unsupervised SAR Anomaly Detection for Marine Oil Spills: Overcoming the Adversarial Over-Reconstruction Trap in Multi-Site Environments"**.

## Repository Structure
- `src/models.py`: PyTorch architecture definitions for SAR-VAE and SAR-AAE.
- `src/evaluate.py`: True pixel-level evaluation script across the stratified 500-patch benchmark.
- `src/generate_grid.py`: Script to generate authentic DARTIS visual grids.
- `dataset_DARTIS_2019/subset_images/`: Directory containing the stratified random sample of 500 Sentinel-1 patches (125 per category: ow, nw, oc, nc).

## Requirements
- Python 3.8+
- PyTorch >= 1.9.0
- torchvision, pandas, numpy, pillow, matplotlib, scikit-learn
