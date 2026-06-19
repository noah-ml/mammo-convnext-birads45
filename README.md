# ConvNeXt-Tiny BI-RADS 4+5 Classifier for VinDr-Mammo

Binary classifier for suspicious mammography findings (BI-RADS 4+5 vs. 1-3) trained on the [VinDr-Mammo](https://vindr.ai/datasets/mammo) dataset.

## Model

- **Architecture:** ConvNeXt-Tiny (ImageNet pretrained, fine-tuned). Code also works with EfficientNet and ResNet (--arch flag).
- **Task:** Binary classification: BI-RADS 4 or 5 (positive) vs. BI-RADS 1-3 (negative)
- **Input:** 1024×384 px grayscale mammograms (replicated to 3 channels), normalized to [0, 1]. The script is resolution-agnostic: it works with any image size baked into the shards (e.g. 512×512, 768×768). All three architectures use global average pooling before the FC head. Adjust `--batch-size` for your VRAM.
- **Training:** Two-phase schedule: head-only warmup (5 epochs) → full fine-tune with cosine LR decay
- **Imbalance handling:** WeightedRandomSampler (target positive rate 20%) + BCEWithLogitsLoss pos_weight

## Results (VinDr-Mammo official test split)

| Metric | Value |
|---|---|
| AUC-ROC | **0.8385** |
| Average Precision | 0.501 |
| Sensitivity | 0.591 |
| Specificity | 0.922 |
| F1 | 0.382 |
| ECE | 0.257 |
| Brier | 0.128 |

Test set: 4000 images, prevalence 4.95% (198 positives).

## Dataset

Download VinDr-Mammo from [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/). The training script reads from TAR shards: DICOM images must be preprocessed into this format first (not included in this repo).

Expected directory layout:

```
data-dir/
  train/*.tar
  val/*.tar
  test/*.tar
  metadata/
    shard_manifest.csv    # columns: image_id, shard, split, label, manufacturer, density
```

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train:**
```bash
python train.py \
    --data-dir /path/to/vindr_tar_shards \
    --output-dir ./output \
    --arch convnext_tiny \
    --batch-size 48 \
    --epochs 50 \
    --no-tta \
    --wandb-project <your-project>
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--arch` | `resnet18` | Architecture: `resnet18`, `efficientnet_b4`, `convnext_tiny` |
| `--batch-size` | 256 | Batch size (use ~48 for ConvNeXt at 1024×384 on an H100) |
| `--freeze-epochs` | 5 | Phase 1 duration (head-only training) |
| `--lr-backbone` | 3e-4 | Backbone learning rate (phase 2) |
| `--lr-head` | 1e-3 | Head learning rate |
| `--dropout` | 0.3 | Dropout before FC head |
| `--oversample-rate` | 0.20 | Target positive rate for oversampling |
| `--patience` | 15 | Early stopping patience (val AUC) |
| `--no-tta` | - | Disable test-time augmentation |
| `--wandb-offline` | - | Log W&B offline, sync later with `wandb sync` |

## Augmentation

Mammo-net style: `ColorJitter(brightness=0.2, contrast=0.2)` and `RandomAffine(degrees=10, scale=(0.9, 1.1))`, each applied with p=0.5. No horizontal flips (all images are laterality-normalized to face left).

## Requirements

```
torch torchvision numpy pandas scikit-learn matplotlib tqdm wandb
```
