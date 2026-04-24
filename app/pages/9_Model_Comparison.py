import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = _ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import CHART_SURFACE, FAVICON_PATH, apply_glass_style, render_page_header

RESULTS_PATH = _ROOT / "results" / "model_comparison_results.csv"


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing comparison file: {RESULTS_PATH}. Run train_compare_models.py first."
        )

    df = pd.read_csv(RESULTS_PATH)

    expected_cols = {
        "model_name",
        "best_val_accuracy_pct",
        "test_accuracy_pct",
        "test_macro_f1",
        "test_weighted_f1",
        "test_top3_accuracy_pct",
        "training_time_sec",
        "saved_model_size_mb",
        "avg_inference_ms_per_image",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Comparison CSV missing columns: {sorted(missing)}")

    return df


def _winner_badge(label: str, value: str, color: str = "#14b8a6") -> str:
    return f"""
    <div style="
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-left: 4px solid {color};
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 0.75rem;
    ">
        <div style="color:#94a3b8;font-size:0.82rem;margin-bottom:0.25rem;">{label}</div>
        <div style="color:#f8fafc;font-size:1.22rem;font-weight:700;">{value}</div>
    </div>
    """


st.set_page_config(
    page_title="Model Comparison",
    page_icon=FAVICON_PATH,
    layout="wide",
)

apply_glass_style(st)

render_page_header(
    st,
    "Model Comparison",
    "Compare deep learning models used for food classification in Calorify AI.",
    kicker="Evaluation",
)

try:
    df = load_results()
except Exception as e:
    st.error(str(e))
    st.stop()

sort_metric = st.selectbox(
    "Sort models by",
    [
        "test_accuracy_pct",
        "test_macro_f1",
        "test_top3_accuracy_pct",
        "training_time_sec",
        "saved_model_size_mb",
        "avg_inference_ms_per_image",
        "best_val_accuracy_pct",
    ],
    index=0,
)

ascending = sort_metric in {
    "training_time_sec",
    "saved_model_size_mb",
    "avg_inference_ms_per_image",
}

df = df.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)

best_acc_model = df.sort_values("test_accuracy_pct", ascending=False).iloc[0]
best_macro_model = df.sort_values("test_macro_f1", ascending=False).iloc[0]
fastest_model = df.sort_values("avg_inference_ms_per_image", ascending=True).iloc[0]
smallest_model = df.sort_values("saved_model_size_mb", ascending=True).iloc[0]

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Best Model Summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Best Accuracy", f'{best_acc_model["model_name"]}')
with c2:
    st.metric("Test Accuracy", f'{best_acc_model["test_accuracy_pct"]:.2f}%')
with c3:
    st.metric("Macro F1", f'{best_acc_model["test_macro_f1"]:.4f}')
with c4:
    st.metric("Top-3 Accuracy", f'{best_acc_model["test_top3_accuracy_pct"]:.2f}%')
st.markdown('</div>', unsafe_allow_html=True)

winner_col1, winner_col2 = st.columns(2)
with winner_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Winners")
    st.markdown(_winner_badge("Highest Accuracy", str(best_acc_model["model_name"])), unsafe_allow_html=True)
    st.markdown(_winner_badge("Best Macro F1", str(best_macro_model["model_name"]), "#6366f1"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with winner_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Efficiency Leaders")
    st.markdown(_winner_badge("Fastest Inference", str(fastest_model["model_name"]), "#f59e0b"), unsafe_allow_html=True)
    st.markdown(_winner_badge("Smallest Model", str(smallest_model["model_name"]), "#e879a8"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Comparison Table")

display_df = df.copy()
display_df = display_df.rename(columns={
    "model_name": "Model",
    "best_val_accuracy_pct": "Best Val Accuracy (%)",
    "test_accuracy_pct": "Test Accuracy (%)",
    "test_macro_f1": "Macro F1",
    "test_weighted_f1": "Weighted F1",
    "test_top3_accuracy_pct": "Top-3 Accuracy (%)",
    "training_time_sec": "Training Time (s)",
    "saved_model_size_mb": "Model Size (MB)",
    "avg_inference_ms_per_image": "Inference Time (ms/image)",
    "trainable_params": "Trainable Params",
    "total_params": "Total Params",
})

st.dataframe(display_df, width="stretch", height=330)
st.markdown('</div>', unsafe_allow_html=True)

charts = [
    ("test_accuracy_pct", "Test Accuracy Comparison", "Accuracy (%)"),
    ("test_macro_f1", "Macro F1 Comparison", "Macro F1"),
    ("training_time_sec", "Training Time Comparison", "Training Time (seconds)"),
    ("saved_model_size_mb", "Model Size Comparison", "Model Size (MB)"),
    ("avg_inference_ms_per_image", "Inference Speed Comparison", "Inference Time (ms/image)"),
    ("test_top3_accuracy_pct", "Top-3 Accuracy Comparison", "Top-3 Accuracy (%)"),
]

for i in range(0, len(charts), 2):
    col1, col2 = st.columns(2)

    metric, title, y_label = charts[i]
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(title)
        fig = px.bar(df, x="model_name", y=metric, text=metric)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CHART_SURFACE,
            plot_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            xaxis_title="Model",
            yaxis_title=y_label,
            height=390,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    if i + 1 < len(charts):
        metric, title, y_label = charts[i + 1]
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(title)
            fig = px.bar(df, x="model_name", y=metric, text=metric)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=CHART_SURFACE,
                plot_bgcolor=CHART_SURFACE,
                font=dict(color="white"),
                xaxis_title="Model",
                yaxis_title=y_label,
                height=390,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Interpretation")

st.write(f"**Highest accuracy:** {best_acc_model['model_name']}")
st.write(f"**Best macro F1:** {best_macro_model['model_name']}")
st.write(f"**Fastest inference:** {fastest_model['model_name']}")
st.write(f"**Smallest model size:** {smallest_model['model_name']}")

st.write(
    "This comparison supports final model selection by balancing classification performance, "
    "efficiency, inference speed, and deployment practicality."
)
st.markdown('</div>', unsafe_allow_html=True)