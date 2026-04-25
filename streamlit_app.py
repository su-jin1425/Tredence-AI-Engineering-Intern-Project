"""
Self-Pruning Neural Network — Streamlit Dashboard
Loads model.pkl generated from case_study.ipynb
"""
import io
import math
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Self-Pruning Neural Network",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; color: #e0e0e0; }
    [data-testid="stSidebar"]          { background: #1a1d2e; }
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2e3155;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 8px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7c6cf8; }
    .metric-label { font-size: 0.85rem; color: #a0a0c0; letter-spacing: 0.05em; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #7c6cf8; margin-top: 1.5rem; margin-bottom: 0.5rem;
    }
    .pill {
        display: inline-block;
        background: #2e3155;
        color: #a0b0ff;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(torch.empty(out_features, in_features))
        self.bias         = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.gate_scores  = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)


class SelfPruningMLP(nn.Module):
    """Inference-only copy of SelfPruningMLP (matches notebook architecture exactly)."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        dims = [3072, 2048, 1024, 512, 256, 10]
        self.layers = nn.ModuleList()
        self.bns    = nn.ModuleList()
        self.drops  = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))
                self.drops.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
        return self.layers[-1](x)


@st.cache_resource
def load_model_artifact(path: str = "model.pkl"):
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    model = SelfPruningMLP(dropout=artifact['config']['dropout'])
    model.load_state_dict(artifact['model_state_dict'])
    model.eval()
    return model, artifact


MODEL_PATH = Path("model.pkl")
if not MODEL_PATH.exists():
    st.error("model.pkl not found. Please run case_study.ipynb first to generate it.")
    st.stop()

model, artifact = load_model_artifact()
CLASSES       = artifact['cifar10_classes']
NORM_MEAN     = artifact['normalize_mean']
NORM_STD      = artifact['normalize_std']
BEST_METRICS  = artifact['metrics']
LAYER_SP      = artifact['layer_sparsity']
LAMBDA_RESULTS = artifact.get('all_lambda_results', [])
BASELINE_ACC   = artifact.get('baseline_accuracy', None)
SWEEP          = artifact.get('threshold_sweep', [])
BONUS_VARIANTS = artifact.get('bonus_variants', [])


def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert('RGB').resize((32, 32), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array(NORM_MEAN)) / np.array(NORM_STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

@torch.no_grad()
def predict(img: Image.Image) -> Tuple[str, float, List[float]]:
    tensor = preprocess_image(img)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().tolist()
    idx    = int(np.argmax(probs))
    return CLASSES[idx], probs[idx] * 100, probs

PLOT_CFG = {
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'text.color':       '#e0e0e0',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'axes.titlecolor':  '#ffffff',
    'grid.color':       '#2a2d4c',
    'grid.alpha':       0.5,
    'axes.grid':        True,
}


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=PLOT_CFG['figure.facecolor'])
    plt.close(fig)
    return buf.getvalue()


def make_confidence_chart(probs: List[float]) -> bytes:
    with plt.rc_context(PLOT_CFG):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = ['#7c6cf8' if v == max(probs) else '#3a3d6c' for v in probs]
        bars = ax.barh(CLASSES, [p * 100 for p in probs], color=colors)
        ax.bar_label(bars, fmt='%.1f%%', padding=4, color='#e0e0e0', fontsize=9)
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Class Confidence Scores')
        ax.set_xlim(0, 110)
    return fig_to_bytes(fig)


def make_gate_histogram() -> bytes:
    with plt.rc_context(PLOT_CFG):
        fig, ax = plt.subplots(figsize=(8, 4))
        all_gates = torch.cat([
            l.get_gates().flatten()
            for l in model.layers
        ]).numpy()
        ax.hist(all_gates, bins=80, color='#7c6cf8', edgecolor='none', alpha=0.85)
        ax.axvline(x=0.01, color='#f85c6c', linestyle='--', linewidth=2, label='Threshold=0.01')
        ax.set_xlabel('Gate Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Final Gate Values')
        ax.legend()
    return fig_to_bytes(fig)


def make_layerwise_sparsity_chart() -> bytes:
    with plt.rc_context(PLOT_CFG):
        fig, ax = plt.subplots(figsize=(9, 4))
        names  = list(LAYER_SP.keys())
        values = list(LAYER_SP.values())
        bars = ax.bar(names, values, color='#7c6cf8', edgecolor='none')
        ax.bar_label(bars, fmt='%.1f%%', padding=4, color='#e0e0e0', fontsize=9)
        ax.set_ylabel('Sparsity (%)')
        ax.set_title('Layer-wise Sparsity')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
        ax.set_ylim(0, 100)
    return fig_to_bytes(fig)


def make_tradeoff_chart() -> bytes:
    if not LAMBDA_RESULTS:
        return None
    with plt.rc_context(PLOT_CFG):
        fig, ax = plt.subplots(figsize=(8, 5))
        sp  = [r['sparsity_pct'] for r in LAMBDA_RESULTS]
        acc = [r['test_accuracy'] for r in LAMBDA_RESULTS]
        lam = [r['lambda'] for r in LAMBDA_RESULTS]
        sc  = ax.scatter(sp, acc, c=np.log10(lam), cmap='plasma', s=120, zorder=5)
        ax.plot(sp, acc, '--', color='#888', linewidth=1, alpha=0.5)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label('log₁₀(λ)')
        for s, a, l in zip(sp, acc, lam):
            ax.annotate(f'λ={l}', (s, a), xytext=(5, 4), textcoords='offset points', fontsize=8)
        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Accuracy vs. Sparsity Tradeoff')
    return fig_to_bytes(fig)


with st.sidebar:
    st.markdown("## Self-Pruning NN")
    st.markdown("**Tredence AI Engineering Internship**")
    st.markdown("---")

    st.markdown("### Best Model")
    st.markdown(f"<span class='pill'>λ = {BEST_METRICS['lambda']}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>Acc = {BEST_METRICS['test_accuracy']:.2f}%</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>Sparsity = {BEST_METRICS['sparsity_pct']:.2f}%</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>{BEST_METRICS['compression_ratio']:.2f}x compressed</span>", unsafe_allow_html=True)

    if BASELINE_ACC:
        st.markdown("---")
        st.markdown("### Dense Baseline")
        st.markdown(f"<span class='pill'>Acc = {BASELINE_ACC:.2f}%</span>", unsafe_allow_html=True)
        acc_drop = BASELINE_ACC - BEST_METRICS['test_accuracy']
        st.markdown(f"<span class='pill'>Drop = {acc_drop:.2f}%</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("3072 → 2048 → 1024 → 512 → 256 → 10")
    st.markdown("PrunableLinear + BatchNorm + ReLU + Dropout")


st.markdown("# Self-Pruning Neural Network")
st.markdown("##### CIFAR-10 Classification with Learnable Weight Gates")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction", "Model Statistics", "Experiment Results", "Pruning Analysis"
])


with tab1:
    col_upload, col_result = st.columns([1, 1.5])

    with col_upload:
        st.markdown("#### Upload an Image")
        st.caption("Upload any 32×32 (or larger) image; it will be resized to CIFAR-10 format.")
        uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp", "webp"])

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)

    with col_result:
        if uploaded:
            img = Image.open(uploaded)
            predicted_class, confidence, probs = predict(img)

            st.markdown("#### Prediction")
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{predicted_class.upper()}</div>
                <div class='metric-label'>PREDICTED CLASS</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{confidence:.1f}%</div>
                <div class='metric-label'>CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence Scores")
            chart_bytes = make_confidence_chart(probs)
            st.image(chart_bytes, use_container_width=True)
        else:
            st.info("Upload an image on the left to run inference.")

    st.markdown("---")
    st.markdown("#### CIFAR-10 Classes")
    cols = st.columns(5)
    for i, cls in enumerate(CLASSES):
        cols[i % 5].markdown(f"<span class='pill'>{i}: {cls}</span>", unsafe_allow_html=True)


with tab2:
    st.markdown("#### Best Model Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{BEST_METRICS['test_accuracy']:.2f}%</div>
        <div class='metric-label'>TEST ACCURACY</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{BEST_METRICS['sparsity_pct']:.2f}%</div>
        <div class='metric-label'>SPARSITY</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{BEST_METRICS['compression_ratio']:.2f}x</div>
        <div class='metric-label'>COMPRESSION RATIO</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{BEST_METRICS['flops_saved_pct']:.2f}%</div>
        <div class='metric-label'>FLOPs SAVED</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Gate Value Histogram")
        st.image(make_gate_histogram(), use_container_width=True)

    with right_col:
        st.markdown("#### Layer-wise Sparsity")
        st.image(make_layerwise_sparsity_chart(), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Parameter Summary")
    pcols = st.columns(3)
    pcols[0].metric("Total Gate Params",  f"{BEST_METRICS['total_weights']:,}")
    pcols[1].metric("Active Params",      f"{BEST_METRICS['active_weights']:,}")
    pcols[2].metric("Params Saved",       f"{BEST_METRICS['params_saved']:,}")

    if SWEEP:
        st.markdown("---")
        st.markdown("#### Threshold Sweep")
        import pandas as pd
        st.dataframe(
            pd.DataFrame(SWEEP).rename(columns={
                'threshold': 'Gate Threshold',
                'accuracy_pct': 'Accuracy (%)',
                'sparsity_pct': 'Sparsity (%)',
            }),
            use_container_width=True,
        )


with tab3:
    if LAMBDA_RESULTS:
        import pandas as pd

        st.markdown("#### Lambda Experiment Results")
        df = pd.DataFrame(LAMBDA_RESULTS)
        if BASELINE_ACC:
            df['accuracy_drop_vs_dense'] = round(BASELINE_ACC - df['test_accuracy'], 3)

        st.dataframe(
            df.rename(columns={
                'lambda': 'Lambda',
                'test_accuracy': 'Test Acc (%)',
                'sparsity_pct': 'Sparsity (%)',
                'active_weights': 'Active Weights',
                'compression_ratio': 'Compression',
                'params_saved': 'Params Saved',
                'flops_saved_pct': 'FLOPs Saved (%)',
                'accuracy_drop_vs_dense': 'Acc Drop',
            }),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### Accuracy vs. Sparsity Tradeoff")
        tradeoff = make_tradeoff_chart()
        if tradeoff:
            st.image(tradeoff, use_container_width=True)

        if BONUS_VARIANTS:
            st.markdown("---")
            st.markdown("#### Bonus Model Variants")
            st.dataframe(pd.DataFrame(BONUS_VARIANTS), use_container_width=True)
    else:
        st.info("Run all lambda experiments in the notebook to populate this tab.")


with tab4:
    st.markdown("#### How Self-Pruning Works")
    st.markdown("""
    Each weight in every `PrunableLinear` layer has a learnable **gate score** parameter.  
    During forward pass:
    ```
    gate = sigmoid(gate_score / temperature)
    effective_weight = weight × gate
    ```
    The sparsity loss penalises the sum of all gate values, pushing them toward **0** (connection removed).  
    Total loss = CrossEntropyLoss + λ × Σ sigmoid(gate\_scores)

    **Temperature annealing** sharpens gates from soft (≈0.5) toward hard binary (0 or 1) over training epochs,  
    using cosine decay from `temperature_start=1.0` → `temperature_end=0.1`.
    """)

    st.markdown("---")
    st.markdown("#### Live Gate Sample (Layer 0, first 8×8 block)")
    with torch.no_grad():
        sample_gates = model.layers[0].get_gates()
    block = sample_gates[:8, :8].numpy()

    with plt.rc_context(PLOT_CFG):
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(block, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, label='Gate Value')
        ax.set_title('Gate Values — Layer 0 (8×8 sample, green=active, red=pruned)')
        ax.set_xlabel('Input Neuron Index')
        ax.set_ylabel('Output Neuron Index')
    st.image(fig_to_bytes(fig), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Active Parameter %  by Layer")
    for name, sparsity_val in LAYER_SP.items():
        active_pct = 100 - sparsity_val
        st.progress(int(active_pct), text=f"{name}: {active_pct:.1f}% active")

    struct_stats = artifact.get('structured_stats', {})
    if struct_stats:
        st.markdown("---")
        st.markdown("#### Structured Pruning — Dead Neurons")
        import pandas as pd
        rows = [{'Layer Metric': k, 'Dead Neurons': v} for k, v in struct_stats.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
