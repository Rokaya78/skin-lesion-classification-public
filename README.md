# Transfer Learning for Skin Lesion Classification
### MobileNetV2 vs. Baseline CNN on HAM10000

A full 4-week project comparing **three transfer-learning strategies** against a from-scratch CNN baseline for melanoma detection on the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

---

## Project Structure

```
skin_lesion_classification/
├── run_all.py                  # Master runner (all 4 weeks)
├── requirements.txt
├── data/                       # Place HAM10000 files here (not in git)
├── checkpoints/                # Saved model weights (not in git)
├── results/                    # JSON reports + plots (not in git)
└── src/
    ├── dataset.py              # HAM10000 loader, augmentation, imbalance handling
    ├── baseline_cnn.py         # Week 1 — VGG-style CNN from scratch (~2.5M params)
    ├── train_baseline.py       # Week 1 — train the scratch baseline
    ├── model.py                # Week 2-3 — MobileNetV2 + custom head, freeze API
    ├── train.py                # Week 2-3 — train any MobileNetV2 strategy
    ├── evaluate.py             # Week 4 — confusion matrix, F1, AUC, BKL-MEL analysis
    └── compare.py              # Week 4 — side-by-side comparison plots & table
```

---

## Timeline

| Week | Task |
|------|------|
| **1** | Data preprocessing (224×224, CLAHE lighting, class weights, stratified split); train **baseline CNN** from scratch |
| **2** | MobileNetV2 **feature extraction** — backbone frozen, only custom head trained |
| **3** | MobileNetV2 **progressive unfreezing** — unfreeze last 5 blocks (epoch 10), then last 10 (epoch 20); data augmentation: rotation, flip, color jitter |
| **4** | **Full fine-tune** + evaluation of all four models; confusion matrices, per-class F1, BKL↔MEL danger analysis, comparison plots |

---

## Models Compared

| Model | Strategy | Pretrained |
|-------|----------|-----------|
| **Baseline CNN** | From scratch (VGG-style blocks) | No |
| **MobileNetV2 — Feature Extraction** | Frozen backbone | ImageNet |
| **MobileNetV2 — Progressive Unfreezing** | Gradual backbone unfreeze | ImageNet |
| **MobileNetV2 — Full Fine-Tune** | All layers trained | ImageNet |

---

## Dataset — HAM10000

7 skin lesion classes:

| Code | Name | Note |
|------|------|------|
| `akiec` | Actinic keratoses | |
| `bcc` | Basal cell carcinoma | |
| `bkl` | **Benign keratosis** | Often confused with melanoma |
| `df` | Dermatofibroma | |
| `mel` | **Melanoma** | Primary target class |
| `nv` | Melanocytic nevi | Most frequent (imbalanced) |
| `vasc` | Vascular lesions | Rarest class |

**Class imbalance** is handled via:
- `WeightedRandomSampler` — oversample rare classes during training
- `CrossEntropyLoss(weight=...)` — inverse-frequency class weights

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download HAM10000 from Harvard Dataverse:
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
# Place files as:
#   data/HAM10000_metadata.csv
#   data/HAM10000_images_part1/
#   data/HAM10000_images_part2/
```

---

## Usage

### Run the full pipeline (all 4 weeks)
```bash
python run_all.py --epochs 30 --batch_size 32
```

### Run individual steps
```bash
# Week 1 — Baseline CNN
python src/train_baseline.py --epochs 40

# Week 2 — Feature extraction
python src/train.py --strategy feature_extraction --epochs 20

# Week 3 — Progressive unfreezing
python src/train.py --strategy progressive --epochs 30 \
    --unfreeze5_epoch 10 --unfreeze10_epoch 20

# Week 3 — Full fine-tune
python src/train.py --strategy full_finetune --epochs 30 --lr 1e-4

# Week 4 — Evaluate each model
python src/evaluate.py --strategy baseline_cnn
python src/evaluate.py --strategy feature_extraction
python src/evaluate.py --strategy progressive
python src/evaluate.py --strategy full_finetune

# Week 4 — Side-by-side comparison
python src/compare.py
```

---

## Output Files (results/)

| File | Description |
|------|-------------|
| `comparison_table.txt` | Accuracy / F1 / AUC table for all 4 models |
| `comparison_overall.png` | Bar chart — overall metrics side by side |
| `comparison_per_class_f1.png` | Per-class F1 for all 4 models |
| `learning_curves.png` | Train/val accuracy + val F1 over epochs |
| `bkl_mel_confusion.png` | BKL ↔ MEL confusion (danger analysis) |
| `{strategy}_confusion_matrix.png` | Per-model confusion matrix |
| `{strategy}_per_class_f1.png` | Per-model class F1 bar chart |
| `{strategy}_eval.json` | Full metrics JSON |

---

## Key Design Decisions

**Progressive unfreezing** uses two separate Adam optimizer instances (rebuilt at each unfreeze step) with a **lower learning rate for backbone layers** (`1e-4`) than the classification head (`1e-3`) to avoid destroying pretrained features.

**BKL ↔ MEL confusion** (`benign keratosis` vs `melanoma`) is explicitly tracked because:
- `mel → bkl` (missed melanoma) is clinically dangerous — false negative
- `bkl → mel` (false alarm) leads to unnecessary procedures

The evaluation script reports both directions and renders a dedicated bar chart.

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.1
- 8 GB RAM (CPU viable — MobileNetV2 is lightweight)
- Optional: CUDA GPU for faster training

---

## References

- Tschandl et al., "The HAM10000 dataset" (2018)
- Sandler et al., ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381) (2018)
- Gumaei et al., ["Comparative Evaluation of Transfer Learning for Brain Tumor Classification"](https://arxiv.org/abs/2310.02270) (2023) — methodology adapted for skin lesions
