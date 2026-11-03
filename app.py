"""
Breast Cancer Diagnosis - Classifier Comparison Dashboard
Pre-trained models are loaded from /model (no retraining). Upload a test CSV
(features + `diagnosis` column) to evaluate and compare the models.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve,
)

# ------------------------------------------------------------------ config
TARGET = "diagnosis"
MODEL_DIR = "model"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "k_nearest_neighbor.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

MODEL_SETTINGS = pd.DataFrame(
    {
        "Model": list(MODEL_FILES.keys()),
        "Preprocessing": ["StandardScaler", "None", "StandardScaler", "None", "StandardScaler"],
        "Key settings": [
            "max_iter=2000",
            "max_depth=5, min_samples_leaf=3",
            "k=7, distance-weighted",
            "Gaussian, default priors",
            "RBF kernel, C=2, gamma='scale', probability=True",
        ],
        "Why it was chosen": [
            "Fast, interpretable linear baseline",
            "Rule-based, easy to explain to non-experts",
            "Instance-based, captures local structure",
            "Probabilistic, very fast to train",
            "Strong non-linear margin-based classifier",
        ],
    }
)

PAGES = [
    "Overview",
    "Evaluation metrics and Classification report",
    "Compare models",
    "Dataset and methodology",
]

st.set_page_config(
    page_title="Breast Cancer Classifier Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(120, 130, 160, 0.10);
        border: 1px solid rgba(120, 130, 160, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers
@st.cache_resource
def load_model(name: str):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Full dataset for the 'Dataset and methodology' page."""
    path = os.path.join("data", "breast_cancer_full.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    raw = load_breast_cancer(as_frame=True).frame.copy()
    raw[TARGET] = 1 - raw["target"]
    return raw.drop(columns=["target"])


@st.cache_data
def load_training_metrics():
    path = os.path.join(MODEL_DIR, "metrics.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def feature_columns(model, data: pd.DataFrame):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        cols = [c for c in data.columns if c != TARGET]
    return list(cols)


def compute_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def validate_upload(uploaded):
    """Return (data, X, y) or stop the page with a clear message."""
    data = pd.read_csv(uploaded)
    if TARGET not in data.columns:
        st.error(f"The uploaded file must contain the target column `{TARGET}`.")
        st.stop()
    first_model = load_model(list(MODEL_FILES.keys())[0])
    cols = feature_columns(first_model, data)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    if data[cols + [TARGET]].isnull().any().any():
        st.error("The uploaded file contains missing values. Please clean it first.")
        st.stop()
    if not set(data[TARGET].unique()).issubset({0, 1}):
        st.error(f"`{TARGET}` must contain only 0 (benign) and 1 (malignant).")
        st.stop()
    if data[TARGET].nunique() < 2:
        st.error("The target column has only one class, so AUC cannot be computed. "
                 "Upload a file containing both benign and malignant rows.")
        st.stop()
    return data, data[cols], data[TARGET].astype(int)


def need_upload_notice():
    st.info(
        "📂 **No test data yet.** Upload a CSV from the sidebar to see results on this page. "
        "The file must contain all 30 feature columns and the `diagnosis` column "
        "(use the sample `test_data.csv` from the sidebar if you need one)."
    )


def draw_confusion(cm, normalize: bool):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    if normalize:
        shown = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(shown, annot=True, fmt=".2%", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🔬 Tumour Classifier Lab")
st.sidebar.caption("Breast Cancer Wisconsin (Diagnostic) | 5 ML models comparison")

st.sidebar.subheader("Name: Shanmuganathan J")

st.sidebar.subheader("Project details")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.subheader("Test data")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button(
            "⬇️ Download sample test_data.csv", f,
            file_name="test_data.csv", mime="text/csv",
        )

model_name = list(MODEL_FILES.keys())[0]
if page == PAGES[1]:
    st.sidebar.divider()
    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

st.sidebar.divider()
st.sidebar.caption("Models are loaded from disk. Nothing is retrained on uploaded data.")


# =========================================================== PAGE: Overview
if page == PAGES[0]:
    st.title("Breast Cancer Diagnosis: Classifier Comparison")
    st.markdown(
        "An interactive dashboard that compares **five machine learning classifiers** "
        "on the task of predicting whether a breast tumour is **benign** or **malignant** "
        "from measurements of cell nuclei."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instances", "569")
    c2.metric("Features", "30")
    c3.metric("Classes", "2")
    c4.metric("Models compared", "5")

    st.subheader("Problem statement")
    st.write(
        "Fine-needle aspirate (FNA) images of a breast mass are digitised and summarised as "
        "numeric features describing the cell nuclei. The goal is to automatically classify "
        "each case as benign or malignant. In a screening setting a **missed malignant case "
        "(false negative)** is far more costly than a false alarm, so recall and MCC are "
        "tracked alongside accuracy."
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("How to use this app")
        st.markdown(
            "1. **Upload** a test CSV in the sidebar (features + `diagnosis`).\n"
            "2. Open **Evaluation metrics and Classification report**, pick a model and "
            "review its metrics, confusion matrix and ROC curve.\n"
            "3. Open **Compare models** to rank all five models on your data.\n"
            "4. Open **Dataset and methodology** for the data description, preprocessing, "
            "model settings and metric definitions."
        )
    with right:
        st.subheader("Models included")
        for m in MODEL_FILES:
            st.markdown(f"- {m}")

    st.subheader("Held-out test results (from training)")
    tm = load_training_metrics()
    if tm is not None:
        best = tm.sort_values("MCC", ascending=False).iloc[0]
        st.success(
            f"Best model on the original held-out test split: **{best['ML Model Name']}** "
            f"(Accuracy {best['Accuracy']:.4f}, Recall {best['Recall']:.4f}, MCC {best['MCC']:.4f})."
        )
        st.dataframe(tm.set_index("ML Model Name"))
    else:
        st.info("Training metrics file not found (model/metrics.csv).")
    st.caption("Positive class = malignant (1). Upload your own test CSV to recompute these live.")


# ===================================== PAGE: Evaluation metrics + Class report
elif page == PAGES[1]:
    st.title("Evaluation Metrics and Classification Report")
    if uploaded is None:
        need_upload_notice()
        st.stop()

    data, X, y = validate_upload(uploaded)
    model = load_model(model_name)
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(model, X, y)

    st.subheader(f"Selected model: {model_name}")
    a, b, c = st.columns(3)
    a.metric("Rows evaluated", len(data))
    b.metric("Malignant share", f"{y.mean():.1%}")
    c.metric("Features used", X.shape[1])

    st.markdown("#### Evaluation metrics")
    cols = st.columns(6)
    for col, name in zip(cols, METRIC_NAMES):
        col.metric(name, f"{metrics[name]:.4f}")

    st.markdown("#### Confusion matrix and ROC curve")
    left, right = st.columns(2)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    with left:
        normalize = st.checkbox("Show as % of each actual class", value=False)
        st.pyplot(draw_confusion(cm, normalize))
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"True negatives: **{tn}** | False positives (false alarms): **{fp}** | "
            f"False negatives (missed malignant): **{fn}** | True positives: **{tp}**"
        )
    with right:
        st.write("")
        st.write("")
        fpr, tpr, _ = roc_curve(y, prob)
        fig2, ax2 = plt.subplots(figsize=(4.8, 4))
        ax2.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {metrics['AUC']:.3f})")
        ax2.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### Classification report")
    report = classification_report(
        y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3))

    with st.expander("Interpretation guide"):
        st.markdown(
            "- **Precision**: of the cases predicted malignant, how many truly were.\n"
            "- **Recall**: of all truly malignant cases, how many were caught.\n"
            "- **F1**: balance of precision and recall.\n"
            "- **support**: number of rows of that class in the uploaded file."
        )

    st.markdown("#### Predictions")
    out = data.copy()
    out["predicted_diagnosis"] = pred
    out["malignant_probability"] = prob.round(4)
    out["correct"] = out["predicted_diagnosis"] == out[TARGET]
    show_wrong = st.checkbox("Show only misclassified rows", value=False)
    view = out[~out["correct"]] if show_wrong else out
    st.dataframe(view[[TARGET, "predicted_diagnosis", "malignant_probability", "correct"]])
    st.download_button(
        "⬇️ Download predictions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ========================================================= PAGE: Compare models
elif page == PAGES[2]:
    st.title("Compare Models")
    if uploaded is None:
        need_upload_notice()
        st.stop()

    data, X, y = validate_upload(uploaded)
    rows = []
    for name in MODEL_FILES:
        rows.append({"ML Model Name": name, **compute_metrics(load_model(name), X, y)})
    table = pd.DataFrame(rows).set_index("ML Model Name")

    best_overall = table["MCC"].idxmax()
    st.success(
        f"Best overall model on your uploaded data: **{best_overall}** "
        f"(MCC {table.loc[best_overall, 'MCC']:.4f}, Accuracy {table.loc[best_overall, 'Accuracy']:.4f})."
    )

    st.markdown("#### Metrics for all five models")
    st.dataframe(table.round(4).style.highlight_max(axis=0, color="#cfe8cf"))
    st.caption("Green = best value in each metric column.")

    st.markdown("#### Best model per metric")
    winners = pd.DataFrame(
        {
            "Metric": METRIC_NAMES,
            "Best model": [table[m].idxmax() for m in METRIC_NAMES],
            "Value": [round(table[m].max(), 4) for m in METRIC_NAMES],
        }
    )
    st.dataframe(winners.set_index("Metric"))

    st.markdown("#### Visual comparison")
    left, right = st.columns(2)
    with left:
        metric_pick = st.selectbox("Single metric", METRIC_NAMES, index=0)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        vals = table[metric_pick].sort_values()
        ax.barh(vals.index, vals.values, color="#4c78a8")
        ax.set_xlim(max(0.0, vals.min() - 0.05), 1.0)
        ax.set_xlabel(metric_pick)
        for i, v in enumerate(vals.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.write("All metrics")
        fig2, ax2 = plt.subplots(figsize=(6, 3.9))
        table[METRIC_NAMES].plot.bar(ax=ax2, width=0.8)
        ax2.set_ylim(max(0.0, table[METRIC_NAMES].min().min() - 0.05), 1.02)
        ax2.set_xlabel("")
        ax2.legend(ncol=3, fontsize=7, loc="lower center")
        plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### ROC curves")
    fig3, ax3 = plt.subplots(figsize=(7, 4.2))
    for name in MODEL_FILES:
        prob = load_model(name).predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        ax3.plot(fpr, tpr, lw=1.8, label=f"{name} ({table.loc[name, 'AUC']:.3f})")
    ax3.plot([0, 1], [0, 1], "--", color="grey")
    ax3.set_xlabel("False positive rate")
    ax3.set_ylabel("True positive rate")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)

    st.download_button(
        "⬇️ Download comparison table (CSV)",
        table.round(4).to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )


# ================================================ PAGE: Dataset and methodology
else:
    st.title("Dataset and Methodology")

    ref = load_reference_data()
    feats = [c for c in ref.columns if c != TARGET]

    st.subheader("Dataset description")
    a, b, c, d = st.columns(4)
    a.metric("Instances", len(ref))
    b.metric("Features", len(feats))
    c.metric("Benign (0)", int((ref[TARGET] == 0).sum()))
    d.metric("Malignant (1)", int((ref[TARGET] == 1).sum()))
    st.write(
        "**Breast Cancer Wisconsin (Diagnostic)**, from the UCI Machine Learning Repository. "
        "Features are computed from digitised images of a fine-needle aspirate of a breast mass. "
        "For each of ten cell-nucleus properties the **mean**, **standard error** and **worst "
        "(largest)** value are recorded, giving 30 numeric features. There are no missing values."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Nucleus property": [
                    "radius", "texture", "perimeter", "area", "smoothness",
                    "compactness", "concavity", "concave points", "symmetry",
                    "fractal dimension",
                ],
                "Meaning": [
                    "Mean distance from centre to the boundary",
                    "Standard deviation of grey-scale values",
                    "Length of the nucleus boundary",
                    "Area of the nucleus",
                    "Local variation in radius lengths",
                    "perimeter^2 / area - 1",
                    "Severity of concave portions of the contour",
                    "Number of concave portions of the contour",
                    "Symmetry of the nucleus",
                    "Coastline approximation - 1",
                ],
            }
        ).set_index("Nucleus property")
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Class distribution")
        counts = ref[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        bars = ax.bar(["Benign (0)", "Malignant (1)"], counts.values,
                      color=["#4c78a8", "#e45756"])
        for bar_, v in zip(bars, counts.values):
            ax.text(bar_.get_x() + bar_.get_width() / 2, v, f"{v}\n({v/len(ref):.1%})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.set_ylabel("Rows")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.markdown("#### Correlation of the 'mean' features")
        mean_cols = [c for c in feats if c.startswith("mean ")]
        fig2, ax2 = plt.subplots(figsize=(5.4, 4.4))
        sns.heatmap(ref[mean_cols].corr(), cmap="coolwarm", center=0, ax=ax2,
                    xticklabels=[c.replace("mean ", "") for c in mean_cols],
                    yticklabels=[c.replace("mean ", "") for c in mean_cols],
                    cbar_kws={"shrink": 0.8})
        plt.setp(ax2.get_xticklabels(), fontsize=7)
        plt.setp(ax2.get_yticklabels(), fontsize=7)
        fig2.tight_layout()
        st.pyplot(fig2)
    st.caption(
        "Radius, perimeter and area are almost perfectly correlated. This violates the "
        "independence assumption of Naive Bayes and helps explain its lower scores."
    )

    with st.expander("Preview the full dataset"):
        st.dataframe(ref.head(25))
        st.dataframe(ref.describe().T.round(3))

    if uploaded is not None:
        up = pd.read_csv(uploaded)
        st.subheader("Your uploaded file")
        u1, u2, u3 = st.columns(3)
        u1.metric("Rows", len(up))
        u2.metric("Columns", up.shape[1])
        if TARGET in up.columns:
            u3.metric("Malignant share", f"{up[TARGET].mean():.1%}")
        st.dataframe(up.head(10))

    st.subheader("Methodology")
    st.markdown(
        "1. **Target coding**: `diagnosis` = 1 for malignant, 0 for benign, so malignant is the "
        "positive class for precision, recall and F1.\n"
        "2. **Split**: 80% train / 20% test, stratified to preserve class balance "
        "(`random_state = 2026`). The 20% test portion is saved as `test_data.csv`.\n"
        "3. **Preprocessing**: standardisation (zero mean, unit variance) for the scale-sensitive "
        "models (Logistic Regression, KNN, SVM). It is saved **inside** each model pipeline, so "
        "uploaded data is scaled exactly as in training.\n"
        "4. **Training**: all five models are fit on the same training split by `train_models.py`.\n"
        "5. **Deployment**: the trained pipelines are stored as `.joblib` files; this app only "
        "loads them and predicts. It never retrains on uploaded data."
    )
    st.markdown("#### Model settings")
    st.dataframe(MODEL_SETTINGS.set_index("Model"))

    st.markdown("#### Evaluation metrics")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": METRIC_NAMES,
                "What it tells you": [
                    "Share of all predictions that are correct",
                    "Ability to rank malignant above benign across all thresholds (1.0 = perfect)",
                    "Of cases flagged malignant, the share that truly are",
                    "Of truly malignant cases, the share that were caught",
                    "Harmonic mean of precision and recall",
                    "Single balanced score from -1 to +1 that uses all four confusion-matrix cells; "
                    "robust to class imbalance",
                ],
            }
        ).set_index("Metric")
    )
