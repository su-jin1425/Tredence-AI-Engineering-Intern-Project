import io
import math
import pickle
import urllib.request
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
        self.temperature: float = 1.0
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        effective_weight = (self.weight * gates).contiguous()
        return F.linear(x, effective_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores / self.temperature)

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'temperature={self.temperature:.4f}')


class SelfPruningMLP(nn.Module):
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
        x = x.reshape(x.size(0), -1).contiguous()
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.drops[i](x)
        return self.layers[-1](x)

    def set_temperature(self, t: float) -> None:
        for layer in self.layers:
            layer.temperature = t

    def apply_hard_pruning(self, threshold: float = 1e-2) -> 'SelfPruningMLP':
        import copy
        hard = copy.deepcopy(self)
        with torch.no_grad():
            for layer in hard.layers:
                mask = (layer.get_gates() >= threshold).float()
                layer.weight.data *= mask
                layer.gate_scores.data = torch.where(
                    mask.bool(),
                    torch.full_like(layer.gate_scores,  10.0),
                    torch.full_like(layer.gate_scores, -10.0),
                )
        return hard

    def get_total_sparsity(self, threshold: float = 1e-2) -> float:
        all_gates = torch.cat([l.get_gates().flatten() for l in self.layers])
        return (all_gates < threshold).float().mean().item()

    def get_active_param_count(self, threshold: float = 1e-2) -> int:
        return sum((l.get_gates() >= threshold).sum().item() for l in self.layers)

    def get_total_gate_count(self) -> int:
        return sum(l.gate_scores.numel() for l in self.layers)


def _load_url_image(url: str, size: int = 64) -> Image.Image:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return _placeholder_solid((120, 120, 120), size)


def _placeholder_solid(color: Tuple[int, int, int], size: int = 64) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), color, dtype=np.uint8))


def _placeholder_automobile(size: int = 64) -> Image.Image:
    arr = np.full((size, size, 3), [30, 30, 30], dtype=np.uint8)
    body_top, body_bot = size // 3, 2 * size // 3
    arr[body_top:body_bot, size // 8: 7 * size // 8] = [180, 50, 50]
    arr[body_bot - size // 10: body_bot, size // 8: 7 * size // 8] = [80, 80, 80]
    wheel_r = size // 8
    for cx in [size // 4, 3 * size // 4]:
        cy = body_bot
        for i in range(size):
            for j in range(size):
                if (i - cy) ** 2 + (j - cx) ** 2 <= wheel_r ** 2:
                    arr[i, j] = [40, 40, 40]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _placeholder_deer(size: int = 64) -> Image.Image:
    arr = np.full((size, size, 3), [34, 120, 34], dtype=np.uint8)
    body_r = size // 5
    bx, by = size // 2, size // 2 + size // 10
    for i in range(size):
        for j in range(size):
            if (i - by) ** 2 + (j - bx) ** 2 <= body_r ** 2:
                arr[i, j] = [160, 100, 60]
    head_r = size // 9
    hx, hy = size // 2, by - body_r - head_r + 2
    for i in range(size):
        for j in range(size):
            if (i - hy) ** 2 + (j - hx) ** 2 <= head_r ** 2:
                arr[i, j] = [160, 100, 60]
    for dx in [-size // 10, size // 10]:
        for dy in range(size // 6):
            r, c = hy - dy, hx + dx
            if 0 <= r < size and 0 <= c < size:
                arr[r, c] = [100, 70, 30]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _placeholder_dog(size: int = 64) -> Image.Image:
    arr = np.full((size, size, 3), [210, 180, 140], dtype=np.uint8)
    body_r = size // 5
    bx, by = size // 2, size // 2 + size // 8
    for i in range(size):
        for j in range(size):
            if (i - by) ** 2 + (j - bx) ** 2 <= body_r ** 2:
                arr[i, j] = [180, 130, 80]
    head_r = size // 8
    hx, hy = size // 2, by - body_r - head_r + 4
    for i in range(size):
        for j in range(size):
            if (i - hy) ** 2 + (j - hx) ** 2 <= head_r ** 2:
                arr[i, j] = [180, 130, 80]
    ear_r = size // 10
    for ex in [-size // 7, size // 7]:
        ey = hy - head_r + 2
        for i in range(size):
            for j in range(size):
                if (i - ey) ** 2 + (j - hx + ex) ** 2 <= ear_r ** 2:
                    arr[i, j] = [120, 80, 40]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _placeholder_ship(size: int = 64) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        t = i / size
        arr[i, :] = [
            int(70 * (1 - t) + 20 * t),
            int(130 * (1 - t) + 60 * t),
            int(200 * (1 - t) + 120 * t),
        ]
    hull_top = size // 2
    for i in range(hull_top, hull_top + size // 5):
        width = int((size * 0.8) * (1 - (i - hull_top) / (size // 5)) + size * 0.2)
        left = (size - width) // 2
        arr[i, left:left + width] = [180, 50, 50]
    for i in range(size // 5, hull_top):
        arr[i, size // 2 - 2: size // 2 + 2] = [220, 220, 200]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _placeholder_truck(size: int = 64) -> Image.Image:
    arr = np.full((size, size, 3), [30, 30, 30], dtype=np.uint8)
    arr[size // 3: 2 * size // 3, size // 8: size // 2] = [60, 90, 180]
    arr[size // 4: 2 * size // 3, size // 2: 7 * size // 8] = [80, 80, 80]
    wheel_r = size // 9
    for cx in [size // 4, size // 2 + size // 6]:
        cy = 2 * size // 3
        for i in range(size):
            for j in range(size):
                if (i - cy) ** 2 + (j - cx) ** 2 <= wheel_r ** 2:
                    arr[i, j] = [40, 40, 40]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


@st.cache_resource
def build_presets() -> Dict[str, Image.Image]:
    airplane_url = "https://raw.githubusercontent.com/su-jin1425/Tredence-AI-Engineering-Intern-Project/266445b81b168615081055dc339c92c3a5f28d1b/Sample%20images/AIRPLANE.png"
    bird_url     = "https://raw.githubusercontent.com/su-jin1425/Tredence-AI-Engineering-Intern-Project/266445b81b168615081055dc339c92c3a5f28d1b/Sample%20images/BIRD.png"
    cat_url      = "https://raw.githubusercontent.com/su-jin1425/Tredence-AI-Engineering-Intern-Project/266445b81b168615081055dc339c92c3a5f28d1b/Sample%20images/CAT.png"
    frog_url     = "https://raw.githubusercontent.com/su-jin1425/Tredence-AI-Engineering-Intern-Project/266445b81b168615081055dc339c92c3a5f28d1b/Sample%20images/FROG.png"
    horse_url    = "https://raw.githubusercontent.com/su-jin1425/Tredence-AI-Engineering-Intern-Project/266445b81b168615081055dc339c92c3a5f28d1b/Sample%20images/HORSE.png"

    return {
        "Airplane":    _load_url_image(airplane_url),
        "Automobile":  _placeholder_automobile(),
        "Bird":        _load_url_image(bird_url),
        "Cat":         _load_url_image(cat_url),
        "Deer":        _placeholder_deer(),
        "Dog":         _placeholder_dog(),
        "Frog":        _load_url_image(frog_url),
        "Horse":       _load_url_image(horse_url),
        "Ship":        _placeholder_ship(),
        "Truck":       _placeholder_truck(),
    }


@st.cache_resource
def load_model_artifact(path: str = "model.pkl"):
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    model = SelfPruningMLP(dropout=artifact['config']['dropout'])
    model.load_state_dict(artifact['model_state_dict'])

    final_temperature = artifact.get('final_temperature', 1.0)
    model.set_temperature(final_temperature)
    model = model.apply_hard_pruning(threshold=1e-2)
    model.set_temperature(1.0)
    model.eval()
    return model, artifact


MODEL_PATH = Path("model.pkl")
if not MODEL_PATH.exists():
    st.error("model.pkl not found. Please run case_study.ipynb first to generate it.")
    st.stop()

model, artifact    = load_model_artifact()
BUILTIN_PRESETS    = build_presets()
CLASSES            = artifact['cifar10_classes']
NORM_MEAN          = artifact['normalize_mean']
NORM_STD           = artifact['normalize_std']
BEST_METRICS       = artifact['metrics']
LAYER_SP           = artifact['layer_sparsity']
LAMBDA_RESULTS     = artifact.get('all_lambda_results', [])
BASELINE_ACC       = artifact.get('baseline_accuracy', None)
SWEEP              = artifact.get('threshold_sweep', [])
BONUS_VARIANTS     = artifact.get('bonus_variants', [])


def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert('RGB').resize((32, 32), Image.LANCZOS)
    arr = np.ascontiguousarray(np.array(img).astype(np.float32) / 255.0)
    arr = ((arr - np.array(NORM_MEAN, dtype=np.float32))
           / np.array(NORM_STD, dtype=np.float32))
    arr = np.ascontiguousarray(arr.astype(np.float32))
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def predict(img: Image.Image) -> Tuple[str, float, List[float]]:
    model.eval()
    tensor = preprocess_image(img)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    idx   = int(np.argmax(probs))
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
            l.get_gates().flatten() for l in model.layers
        ]).numpy()
        ax.hist(all_gates, bins=80, color='#7c6cf8', edgecolor='none', alpha=0.85)
        ax.axvline(x=0.01, color='#f85c6c', linestyle='--', linewidth=2,
                   label='Threshold=0.01')
        ax.set_xlabel('Gate Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Final Gate Values (after hard pruning)')
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


def make_tradeoff_chart() -> Optional[bytes]:
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
        cb.set_label('log10(lambda)')
        for s, a, l in zip(sp, acc, lam):
            ax.annotate(f'lambda={l}', (s, a), xytext=(5, 4),
                        textcoords='offset points', fontsize=8)
        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Accuracy vs. Sparsity Tradeoff')
    return fig_to_bytes(fig)


def make_preset_grid(selected_name: str) -> bytes:
    cols_n = 5
    rows_n = math.ceil(len(BUILTIN_PRESETS) / cols_n)
    with plt.rc_context(PLOT_CFG):
        fig, axes = plt.subplots(rows_n, cols_n,
                                 figsize=(cols_n * 2.2, rows_n * 2.4))
        axes_flat = axes.flatten() if rows_n > 1 else list(axes)
        for idx, (name, img) in enumerate(BUILTIN_PRESETS.items()):
            ax = axes_flat[idx]
            ax.imshow(img.resize((64, 64)))
            ax.set_title(name, fontsize=7, color='#e0e0e0', pad=3)
            ax.axis('off')
            if name == selected_name:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#7c6cf8')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
        for idx in range(len(BUILTIN_PRESETS), len(axes_flat)):
            axes_flat[idx].axis('off')
        fig.tight_layout(pad=0.5)
    return fig_to_bytes(fig)


with st.sidebar:
    st.markdown("## Self-Pruning NN")
    st.markdown("**Tredence AI Engineering Internship**")
    st.markdown("---")

    st.markdown("### Best Model")
    st.markdown(f"<span class='pill'>lambda = {BEST_METRICS['lambda']}</span>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>Acc = {BEST_METRICS['test_accuracy']:.2f}%</span>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>Sparsity = {BEST_METRICS['sparsity_pct']:.2f}%</span>",
                unsafe_allow_html=True)
    st.markdown(
        f"<span class='pill'>{BEST_METRICS['compression_ratio']:.2f}x compressed</span>",
        unsafe_allow_html=True)

    if BASELINE_ACC:
        st.markdown("---")
        st.markdown("### Dense Baseline")
        st.markdown(f"<span class='pill'>Acc = {BASELINE_ACC:.2f}%</span>",
                    unsafe_allow_html=True)
        acc_drop = BASELINE_ACC - BEST_METRICS['test_accuracy']
        st.markdown(f"<span class='pill'>Drop = {acc_drop:.2f}%</span>",
                    unsafe_allow_html=True)

    ft = artifact.get('final_temperature', 1.0)
    st.markdown("---")
    st.markdown("### Training Config")
    st.markdown(f"<span class='pill'>Final Temp = {ft:.4f}</span>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("3072 -> 2048 -> 1024 -> 512 -> 256 -> 10")
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
        st.markdown("#### Upload or Select an Image")
        st.caption("Upload your own image or choose a sample below.")

        input_mode = st.radio("Input mode", ["Upload image", "Use preset sample"],
                              horizontal=True)

        active_image: Optional[Image.Image] = None

        if input_mode == "Upload image":
            uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp", "webp"])
            if uploaded:
                active_image = Image.open(uploaded)
                st.image(active_image, caption="Uploaded Image", use_container_width=True)

        else:
            preset_name = st.selectbox("Choose a preset", list(BUILTIN_PRESETS.keys()))
            active_image = BUILTIN_PRESETS[preset_name]

            st.image(active_image, caption=f"Selected: {preset_name}",
                     use_container_width=False, width=160)

            st.markdown("**All presets:**")
            st.image(make_preset_grid(preset_name), use_container_width=True)
            st.caption(
                "Real images are loaded from the project repo. "
                "Placeholder images are synthetic renderings. "
                "Upload a real photo for the most meaningful CIFAR-10 predictions."
            )

    with col_result:
        if active_image is not None:
            predicted_class, confidence, probs = predict(active_image)

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
            st.image(make_confidence_chart(probs), use_container_width=True)
        else:
            st.info("Upload an image or select a preset on the left to run inference.")

    st.markdown("---")
    st.markdown("#### CIFAR-10 Classes")
    cls_cols = st.columns(5)
    for i, cls in enumerate(CLASSES):
        cls_cols[i % 5].markdown(
            f"<span class='pill'>{i}: {cls}</span>", unsafe_allow_html=True)


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
    pcols[0].metric("Total Gate Params", f"{BEST_METRICS['total_weights']:,}")
    pcols[1].metric("Active Params",     f"{BEST_METRICS['active_weights']:,}")
    pcols[2].metric("Params Saved",      f"{BEST_METRICS['params_saved']:,}")

    if SWEEP:
        st.markdown("---")
        st.markdown("#### Threshold Sweep")
        import pandas as pd
        st.dataframe(
            pd.DataFrame(SWEEP).rename(columns={
                'threshold':    'Gate Threshold',
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
                'lambda':                'Lambda',
                'test_accuracy':         'Test Acc (%)',
                'sparsity_pct':          'Sparsity (%)',
                'active_weights':        'Active Weights',
                'compression_ratio':     'Compression',
                'params_saved':          'Params Saved',
                'flops_saved_pct':       'FLOPs Saved (%)',
                'accuracy_drop_vs_dense':'Acc Drop',
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

    **During forward pass (temperature-annealed):**
    ```
    gate             = sigmoid(gate_score / temperature)
    effective_weight = weight x gate
    ```

    **Total training loss:**
    ```
    Loss = CrossEntropyLoss + lambda x mean(sigmoid(gate_scores / T))
    ```

    The sparsity term pushes **gate_scores negative** (connection removed).

    **Temperature annealing** sharpens gates from soft (T=2.0 -> sigmoid near 0.5 for all)
    toward hard binary (T=0.05 -> sigmoid near 0 or 1) using cosine decay over training epochs.
    At T=0.05 a gate_score of just +/-1.0 gives sigmoid(+/-20) near 0 or 1, making pruning
    easy to achieve even with moderate lambda values.

    **After training**, hard pruning binarises gate_scores to +/-10 for stable inference.
    """)

    st.markdown("---")
    st.markdown("#### Live Gate Sample (Layer 0, first 8x8 block)")
    with torch.no_grad():
        sample_gates = model.layers[0].get_gates()
    block = sample_gates[:8, :8].numpy()

    with plt.rc_context(PLOT_CFG):
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(block, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, label='Gate Value')
        ax.set_title('Gate Values - Layer 0 (8x8 sample, green=active, red=pruned)')
        ax.set_xlabel('Input Neuron Index')
        ax.set_ylabel('Output Neuron Index')
    st.image(fig_to_bytes(fig), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Active Parameter % by Layer")
    for name, sparsity_val in LAYER_SP.items():
        active_pct = 100 - sparsity_val
        st.progress(int(active_pct), text=f"{name}: {active_pct:.1f}% active")

    struct_stats = artifact.get('structured_stats', {})
    if struct_stats:
        st.markdown("---")
        st.markdown("#### Structured Pruning - Dead Neurons")
        import pandas as pd
        rows = [{'Layer Metric': k, 'Dead Neurons': v}
                for k, v in struct_stats.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
