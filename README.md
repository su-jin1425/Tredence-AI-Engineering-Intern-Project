# The Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/su-jin1425/Tredence-AI-Engineering-Intern-Project/blob/main/case_study.ipynb)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://tredence-ai-engineering-intern-project-sujith.streamlit.app)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Checkpoints%20%7C%20Results%20%7C%20Source-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1E4L2oDu_bb3x5E-6Hc4fHo3yqgalDeJe?usp=drive_link)

---

## Overview

This project implements a neural network that **learns to prune itself during training** using learnable per-weight gates. Each connection in every `PrunableLinear` layer is controlled by a gate driven by a temperature-annealed sigmoid:

```
temperature   = cosine_anneal(T_START=5.0 → T_END=0.1)
gates         = sigmoid(gate_scores / temperature)   # element-wise, in [0, 1]
pruned_weight = weight × gates
output        = F.linear(x, pruned_weight, bias)
```

When a gate approaches 0, the connection is effectively removed. The network simultaneously optimises for:

1. **Classification accuracy** — Cross-Entropy Loss with label smoothing = 0.1
2. **Sparsity** — mean of all gate values across all `PrunableLinear` layers, weighted by λ

```
Total Loss = CrossEntropyLoss + λ × mean(sigmoid(gate_scores))
```

> **Architecture:** Pure MLP — `SelfPruningMLP` with dimensions `3072 → 2048 → 1024 → 512 → 256 → 10`. All dense layers are `PrunableLinear`. BatchNorm is placed after gate multiplication for clean gradients. Three bonus variants (CompactMLP, DeepMLP, ResidualMLP) are also trained.

---

## Architecture

### SelfPruningMLP (Main Model)

A 5-hidden-layer MLP trained end-to-end on raw CIFAR-10 pixels (32×32×3 = 3072 inputs):

```
Input (3072)
  └─ PrunableLinear(3072 → 2048) → BN → ReLU → Dropout(0.25)
  └─ PrunableLinear(2048 → 1024) → BN → ReLU → Dropout(0.25)
  └─ PrunableLinear(1024 → 512)  → BN → ReLU → Dropout(0.25)
  └─ PrunableLinear(512  → 256)  → BN → ReLU → Dropout(0.25)
  └─ PrunableLinear(256  → 10)   → logits
```

All layers are `PrunableLinear` — no standard `nn.Linear` in the prunable model. Total parameters: ~9,058,058 (dense baseline).

### PrunableLinear — Core Component

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.temperature = 1.0   # plain float, not a Parameter

    def forward(self, x):
        gates         = torch.sigmoid(self.gate_scores / self.temperature)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def sparsity_penalty(self):
        return torch.sigmoid(self.gate_scores / self.temperature).mean()
```

**Gate initialisation:** `gate_scores = 0` → `sigmoid(0) = 0.5` at start (neutral "half-open" state). The sparsity loss drives unimportant gates toward 0. Weights are initialised with Kaiming uniform; initial values are saved for lottery ticket retraining.

### Temperature Annealing

Gate sharpness is controlled by a cosine-annealed temperature across training:

```
T_START = 5.0  →  T_END = 0.1   (cosine schedule over num_epochs)
```

- High temperature → smooth/soft gates (good for early exploration)
- Low temperature → near-binary gates (enforces hard sparsity structure at end)

### Why Sparsity Emerges

Total loss: `L = CE(logits, y) + λ × mean(sigmoid(gate_scoreᵢ))`

The gradient of the penalty w.r.t. each gate score is always non-negative:

```
∂penalty/∂gate_scoreᵢ = sigmoid(gate_scoreᵢ/T) × (1 − sigmoid(gate_scoreᵢ/T)) / T ≥ 0
```

The optimiser always has an incentive to reduce gate scores. Gates useful for classification resist this pressure (large CE gradient). Unimportant gates get driven to 0. This produces the characteristic **bimodal distribution**: large spike near 0 (pruned) and a cluster near 1 (important connections kept).

---

## Bonus Model Variants

| Variant | Architecture | Test Accuracy | Sparsity |
|---|---|---|---|
| **SelfPruningMLP** | 3072→2048→1024→512→256→10 | **63.06%** | 33.01% |
| **ResidualMLP** | embed(3072→512) + 2 residual blocks + head | **63.72%** | 36.96% |
| **DeepMLP** | 3072→2048→2048→1024→512→256→128→10 | 61.08% | 30.68% |
| **CompactMLP** | 3072→512→256→10 | 59.00% | 40.85% |
| **DenseBaseline** | Same MLP, `nn.Linear`, λ=0 | 62.56% | 0.00% |

**CompactMLP** is a lightweight 3-layer model. **DeepMLP** is 7-layer. **ResidualMLP** uses an embed layer (3072→512) followed by two `ResidualPrunableBlock`s each containing two `PrunableLinear` layers with skip connections. All variants are trained at `best_lam=0.05` for direct comparison.

---

## Lambda Sweep Results

8-experiment sweep over `λ ∈ [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]`:

| λ | Test Acc (%) | Sparsity (%) | Active Weights | Compression | FLOPs Saved (%) |
|---|---|---|---|---|---|
| 0.0001 | 62.77 | 28.07 | 6,507,638 | 1.39× | 28.07% |
| 0.0005 | 62.96 | 28.14 | 6,500,754 | 1.39× | 28.14% |
| 0.001  | 62.74 | 28.81 | 6,440,539 | 1.40× | 28.81% |
| 0.005  | 62.86 | 28.66 | 6,454,090 | 1.40× | 28.66% |
| 0.01   | 62.48 | 29.18 | 6,406,790 | 1.41× | 29.18% |
| **0.05**   | **63.06** | **33.01** | **6,060,610** | **1.49×** | **33.01%** |
| 0.1    | 62.90 | 37.26 | 5,676,031 | 1.59× | 37.26% |
| 0.5    | 57.61 | 55.82 | 3,997,005 | 2.26× | 55.82% |

**Best model: λ=0.05** — 63.06% accuracy with 33% sparsity (1.49× compression).

---

## Hard Pruning & Lottery Ticket

**Hard pruning** (`apply_hard_pruning`): sub-threshold weights are zeroed; gate_scores are binarised to ±10. Accuracy after hard pruning at threshold 0.01: **59.94%**.

**Threshold sweep** on best model:

| Threshold | Accuracy (%) | Sparsity (%) |
|---|---|---|
| 0.100 | 61.00 | 49.98 |
| 0.010 | 59.94 | 33.01 |
| 0.001 | 59.65 | 19.25 |

**Lottery ticket** (`lottery_ticket_reset`): resets weights to their Kaiming-initialised values masked by the winning ticket subnetwork, then retrains for 40 epochs at `best_lam`. Lottery ticket accuracy: **62.15%** — very close to the full model, validating the lottery ticket hypothesis.

**Structured pruning:** Dead neuron counting (all input or all output gates below threshold) per layer. With the best model at threshold=0.01, zero fully dead neurons were found across all 5 layers, indicating distributed rather than structured pruning.

---

## Training Configuration

| Parameter | Value |
|---|---|
| `batch_size` | 256 |
| `num_epochs` | 80 |
| `warmup_epochs` | 5 |
| `patience` (early stopping) | 15 |
| `dropout_rate` | 0.25 |
| `gate_threshold` | 0.01 |
| `lr` | 3×10⁻⁴ |
| `weight_decay` | 1×10⁻⁴ |
| `grad_clip` | 1.0 |
| Label smoothing | 0.1 |
| Optimizer | AdamW |
| LR schedule | Linear warmup + Cosine Annealing |
| Mixed precision | AMP (if CUDA available) |
| Temperature schedule | Cosine: T=5.0 → 0.1 |
| Seed | 42 |

---

## Data Augmentation (CIFAR-10)

```python
train_transforms = [
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    RandomRotation(10),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
]
```

Test set uses only normalisation (no augmentation).

---

## Metrics Computed

For each λ experiment:

| Metric | Description |
|---|---|
| `test_accuracy` | % correct on CIFAR-10 test set (10,000 images) |
| `sparsity_pct` | % of gates below threshold |
| `active_weights` | Weights with gate ≥ threshold |
| `compression_ratio` | total_weights / active_weights |
| `params_saved` | Count of pruned weights |
| `flops_saved_pct` | FLOPs reduction from pruning |
| `accuracy_drop_vs_dense` | Difference from dense baseline (62.56%) |

---

## Output Charts

Six charts are generated and saved to `Results/`:

- **`gate_histogram.png`** — Distribution of final gate values (bimodal: near-0 pruned + near-1 kept)
- **`lambda_accuracy.png`** — Test accuracy vs. λ on log scale
- **`lambda_sparsity.png`** — Sparsity % vs. λ on log scale
- **`tradeoff_curve.png`** — Accuracy vs. sparsity scatter, coloured by log₁₀(λ)
- **`training_loss.png`** — Total loss curve for best model (λ=0.05)
- **`layerwise_sparsity.png`** — Bar chart of sparsity % per layer

---

## Project Structure

```
├── case_study.ipynb           ← Full 16-cell experiment notebook
├── streamlit_app.py           ← Interactive Streamlit dashboard
├── requirements.txt           ← All dependencies
├── model.pkl                  ← Generated by notebook (best model state + metrics)
└── Results/                   ← All output files (capital R)
    ├── experiment_results.csv
    ├── summary_table.csv
    ├── best_model_metrics.json
    ├── threshold_sweep.csv
    ├── gate_histogram.png
    ├── lambda_accuracy.png
    ├── lambda_sparsity.png
    ├── tradeoff_curve.png
    ├── training_loss.png
    └── layerwise_sparsity.png
```

Checkpoints are saved after every epoch to Google Drive at `self_pruning_nn/Checkpoints/` and automatically resumed if interrupted.

---

## model.pkl Contents

```python
{
    'model_state_dict':     ...,          # best model weights (λ=0.05)
    'model_class':          'SelfPruningMLP',
    'config':               {'dropout': 0.25},
    'metrics':              best_result,
    'layer_sparsity':       {...},        # per-layer sparsity at best threshold
    'structured_stats':     {...},        # dead neuron counts per layer
    'threshold_sweep':      [...],        # accuracy at 0.1 / 0.01 / 0.001
    'cifar10_classes':      [...],        # 10 class names
    'normalize_mean':       (0.4914, 0.4822, 0.4465),
    'normalize_std':        (0.2023, 0.1994, 0.2010),
    'all_lambda_results':   [...],        # full 8-λ sweep table
    'baseline_accuracy':    62.56,
    'bonus_variants':       [...],        # CompactMLP, DeepMLP, ResidualMLP results
    'final_temperature':    0.1,
}
```

---

## Google Drive Resources

All training artifacts — checkpoints, result CSVs/PNGs, and source files — are publicly available on Google Drive:

**[📁 Open Drive Folder](https://drive.google.com/drive/folders/1E4L2oDu_bb3x5E-6Hc4fHo3yqgalDeJe?usp=drive_link)**

| Folder / File | Contents |
|---|---|
| `Checkpoints/` | Per-epoch `.pt` files for baseline, all 8 λ runs, lottery ticket, and bonus variants |
| `Results/` | All 6 charts (`.png`), `experiment_results.csv`, `summary_table.csv`, `best_model_metrics.json`, `threshold_sweep.csv` |
| `case_study.ipynb` | Full source notebook (16 cells) |
| `streamlit_app.py` | Streamlit dashboard source |
| `model.pkl` | Exported best model state + all metrics |
| `requirements.txt` | Dependency list |

---

## Setup

**Python 3.10+ recommended. GPU (CUDA) strongly recommended.**

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `streamlit`.

---

## Running the Notebook

**Recommended: Google Colab (free T4 GPU)**

1. Upload `case_study.ipynb` and `requirements.txt` to Colab
2. In the first cell, add: `!pip install -r requirements.txt`
3. Set Runtime → Change runtime type → **T4 GPU**
4. Run all cells top-to-bottom

Training 8 λ experiments × 80 epochs takes ~12–25 hours on a T4 GPU. Checkpoints save after every epoch — if interrupted, simply re-run and training resumes from the last checkpoint.

**Locally (CPU):** Reduce `num_epochs` to 10 for a quick smoke test. Full training will take many hours.

---

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app works in **demo mode** (no `model.pkl` needed) showing representative results. It automatically switches to **real model mode** after the notebook generates `model.pkl`. Live hosted version: [tredence-ai-engineering-intern-project-sujith.streamlit.app](https://tredence-ai-engineering-intern-project-sujith.streamlit.app)

---

## Evaluation Criteria Coverage

| Criterion | Implementation |
|---|---|
| **PrunableLinear** | `gate_scores` same shape as `weight`; forward: `F.linear(x, weight × sigmoid(gate_scores/T), bias)`; both `weight` and `gate_scores` passed to optimizer |
| **Temperature annealing** | Cosine schedule T_START=5.0 → T_END=0.1 propagated to all layers each epoch via `set_temperature()` |
| **Training loop** | `total_loss = CE_loss + λ × model.get_sparsity_penalty()`, where penalty is MEAN of gate values across all layers |
| **Hard pruning** | `apply_hard_pruning()` zeroes sub-threshold weights, binarises gate_scores to ±10; returns a deepcopy |
| **Lottery ticket** | `lottery_ticket_reset()` resets weights to Kaiming-initialised values masked by winning ticket, then retrains |
| **Threshold sweep** | 3 thresholds (0.1, 0.01, 0.001) tested on best model |
| **Structured pruning** | Dead neuron counting (all input or output gates pruned) per layer |
| **Bonus variants** | CompactMLP, DeepMLP, ResidualMLP — all trained at best λ=0.05 |
| **Results quality** | Bimodal gate histogram ✓, λ trade-off table ✓, layerwise sparsity ✓ |
| **Code quality** | Type annotations, docstrings, modular functions, seed=42, early stopping, AMP, checkpoint resumption |