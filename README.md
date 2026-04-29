# Tredence-AI-Engineering-Intern-Project

# The Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study Submission

Here is the development link : [Run on Google Colab](https://colab.research.google.com/github/su-jin1425/Tredence-AI-Engineering-Intern-Project/blob/main/case_study.ipynb)

Here is the Playable link : [Run on Chrome](https://tredence-ai-engineering-intern-project-sujith.streamlit.app)

---

## Project Overview

This project implements a neural network that learns to **prune itself during training** using learnable weight gates. Each connection in every `PrunableLinear` layer is controlled by a gate with a temperature-annealed sigmoid:

```
temperature  = cosine_anneal(T_START → T_END)
gates        = sigmoid(gate_scores / temperature)   # element-wise, in [0, 1]
pruned_weight = weight × gates
output       = F.linear(x, pruned_weight, bias)
```

When a gate approaches 0, the connection is effectively removed. The network simultaneously optimises for:
1. **Classification accuracy** (Cross-Entropy Loss with label smoothing = 0.1)
2. **Sparsity** (mean of all gate values across all `PrunableLinear` layers, weighted by λ)

**Total Loss = CrossEntropyLoss + λ × mean(sigmoid(gate_scores))**

> **Architecture:** Pure MLP — `SelfPruningMLP` with dimensions `3072 → 2048 → 1024 → 512 → 256 → 10`.  
> All dense layers are `PrunableLinear`. BatchNorm is placed after gate multiplication for clean gradients. Three bonus variants (CompactMLP, DeepMLP, ResidualMLP) are also trained.

---

## Architecture: SelfPruningMLP

The main model is a 5-hidden-layer MLP trained end-to-end on raw CIFAR-10 pixels:

```
Input (3072)
  └─ PrunableLinear(3072 → 2048) → BN → ReLU → Dropout
  └─ PrunableLinear(2048 → 1024) → BN → ReLU → Dropout
  └─ PrunableLinear(1024 → 512)  → BN → ReLU → Dropout
  └─ PrunableLinear(512  → 256)  → BN → ReLU → Dropout
  └─ PrunableLinear(256  → 10)   → logits
```

All layers are `PrunableLinear` — no standard `nn.Linear` in the prunable model.

---

## PrunableLinear: Core Component

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
        """Mean of gate values — normalised L1, scale-independent of layer size."""
        return torch.sigmoid(self.gate_scores / self.temperature).mean()
```

**Gate initialisation:** `gate_scores = 0` → `sigmoid(0) = 0.5` at the start (neutral "half-open" state). The sparsity loss drives unimportant gates toward 0.

---

## SparsityLoss Formula

The sparsity penalty is the **mean** of all gate values across all PrunableLinear layers (normalised L1, for training stability):

```python
def get_sparsity_penalty(self) -> torch.Tensor:
    return torch.stack([l.sparsity_penalty() for l in self.layers]).mean()
```

This keeps the penalty scale consistent regardless of model size, making the same λ values meaningful across `CompactMLP`, `SelfPruningMLP`, and `DeepMLP`.

---

## Temperature Annealing

Gate sharpness is controlled by a cosine-annealed temperature:

```
T_START = 5.0  →  T_END = 0.1   (cosine schedule over num_epochs)
```

High temperature → smooth/soft gates (good for early training).  
Low temperature → near-binary gates (enforces hard sparsity structure at end of training).

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
| LR schedule | Warmup + Cosine Annealing |
| Mixed precision | AMP (if CUDA available) |

**Lambda sweep:** `[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]`

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

---

## Features

| Feature | Detail |
|---|---|
| `PrunableLinear` with `gate_scores` | `sigmoid(gate_scores / T) × weight`, gradients flow through both |
| Temperature annealing | Cosine schedule T=5.0 → 0.1 over training |
| SparsityLoss | MEAN of all gate values (normalised L1) |
| 8-λ experiment sweep | `[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]` |
| Dense baseline | Same MLP capacity with `nn.Linear` (λ=0) |
| Hard pruning | Gates < threshold zeroed, gate_scores binarised to ±10 |
| Lottery ticket reset + retrain | Winning ticket subnetwork identified and retrained |
| Threshold sweep (3 thresholds) | 0.1, 0.01, 0.001 |
| Structured pruning analysis | Dead neuron counting per layer (input + output) |
| Bonus variants | `CompactMLP` (3072→512→256→10), `DeepMLP` (7-layer), `ResidualMLP` (residual blocks) |
| 6 result charts | Histogram, λ-accuracy, λ-sparsity, tradeoff, training loss, layerwise |
| Checkpoint resumption | Saves/loads full training state (model, optimizer, scheduler, scaler) |
| AMP support | Mixed precision training on CUDA |
| Streamlit dashboard | Live demo + full analytics, works in demo mode without `model.pkl` |

---

## Bonus Model Variants

### CompactMLP
Lightweight 3-layer model: `3072 → 512 → 256 → 10`

### DeepMLP
7-layer deep model: `3072 → 2048 → 2048 → 1024 → 512 → 256 → 128 → 10`

### ResidualMLP
Residual architecture: embed layer (3072→512) + 2 `ResidualPrunableBlock`s + head. Each block contains two `PrunableLinear` layers with skip connections.

All three are trained at `best_lam` for direct comparison.

---

## Why Sparsity Emerges: Gradient Analysis

The total loss is: `L = CE(logits, y) + λ × mean(sigmoid(gate_scoreᵢ))`

The gradient of the penalty w.r.t. each gate score is:

```
∂penalty/∂gate_scoreᵢ = sigmoid(gate_scoreᵢ/T) × (1 − sigmoid(gate_scoreᵢ/T)) / T ≥ 0
```

This is always **non-negative**, so the optimiser always has an incentive to reduce gate scores. A smaller gate score → smaller sigmoid → gate closer to 0 → weight effectively pruned.

Gates useful for classification resist this pressure (their CE gradient is large). Gates for unimportant connections get driven to 0 by the L1 penalty. This produces the characteristic **bimodal distribution**: large spike near 0 (pruned) and cluster near 1 (important).

---

## Metrics Computed

For each λ experiment:

| Metric | Description |
|---|---|
| `test_accuracy` | % on CIFAR-10 test set |
| `sparsity_pct` | % of gates below threshold |
| `active_weights` | Weights with gate ≥ threshold |
| `compression_ratio` | total / active weights |
| `params_saved` | Pruned weight count |
| `flops_saved_pct` | FLOPs reduction from pruning |
| `accuracy_drop_vs_dense` | Difference from dense baseline |

---

## Project Structure

```
├── case_study.ipynb       ← Full 16-cell experiment notebook
├── streamlit_app.py       ← Interactive Streamlit dashboard
├── requirements.txt       ← All dependencies
├── model.pkl              ← Generated by notebook (best model state + all metrics)
└── Results/               ← Capital R — all output files go here
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

---

## Setup

**Python 3.10+ recommended. GPU (CUDA) strongly recommended for the notebook.**

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

**Recommended: Google Colab (free T4 GPU)**

1. Upload `case_study.ipynb` and `requirements.txt` to Colab
2. In the first cell, add: `!pip install -r requirements.txt`
3. Set Runtime → Change runtime type → **T4 GPU**
4. Run all cells — training 8 λ experiments × 80 epochs takes ~12–25 hours on GPU

**Locally (CPU):** Training will take several hours. Reduce `num_epochs` to 10 for a quick test.

Checkpoints are saved after every epoch to Google Drive (`Checkpoints/`) and automatically resumed if interrupted.

---

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app works in **demo mode** (no `model.pkl` needed) showing representative results, and automatically switches to **real model mode** after the notebook generates `model.pkl`.

---

## model.pkl Contents

The exported artifact includes:

```python
{
    'model_state_dict':     ...,          # best model weights
    'model_class':          'SelfPruningMLP',
    'config':               {'dropout': 0.25},
    'metrics':              best_result,
    'layer_sparsity':       {...},
    'structured_stats':     {...},        # dead neuron counts
    'threshold_sweep':      [...],        # accuracy at 0.1 / 0.01 / 0.001
    'cifar10_classes':      [...],
    'normalize_mean':       (0.4914, 0.4822, 0.4465),
    'normalize_std':        (0.2023, 0.1994, 0.2010),
    'all_lambda_results':   [...],        # full sweep table
    'baseline_accuracy':    ...,
    'bonus_variants':       [...],        # CompactMLP, DeepMLP, ResidualMLP
    'final_temperature':    0.1,
}
```

---

## Evaluation Criteria Coverage

| Criterion | How We Satisfy It |
|---|---|
| **Correct PrunableLinear** | `gate_scores` same shape as `weight`; forward: `F.linear(x, weight × sigmoid(gate_scores/T), bias)`; both `weight` and `gate_scores` in optimizer |
| **Temperature annealing** | Cosine schedule T_START=5.0 → T_END=0.1 propagated to all layers each epoch |
| **Correct Training Loop** | `total_loss = CE_loss + λ × model.get_sparsity_penalty()`, where penalty is MEAN of gate values |
| **Hard pruning** | `apply_hard_pruning()` zeroes sub-threshold weights, binarises gate_scores to ±10 |
| **Lottery ticket** | `lottery_ticket_reset()` resets weights to initial values masked by winning ticket, then retrains |
| **Threshold sweep** | 3 thresholds (0.1, 0.01, 0.001) tested on best model |
| **Structured pruning** | Dead neuron counting (all input or output gates pruned) per layer |
| **Bonus variants** | CompactMLP, DeepMLP, ResidualMLP — all trained at best λ |
| **Quality of Results** | Bimodal gate histogram ✓, λ trade-off table ✓, layerwise sparsity ✓ |
| **Code Quality** | Type annotations, docstrings, modular functions, seed=42, early stopping, AMP, checkpoint resumption |
