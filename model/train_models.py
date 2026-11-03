"""
train_models.py
---------------
Trains five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
downloaded directly from the UCI Machine Learning Repository:
    https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
evaluates them on a held-out test split, and saves everything the Streamlit
app needs:

    model/*.joblib      -> one saved pipeline per classifier
    model/metrics.csv   -> comparison table (used in README / app)
    test_data.csv       -> held-out test rows (features + target) for the app
    data/wdbc.data              -> raw file downloaded from UCI (cached copy)
    data/breast_cancer_full.csv -> the complete cleaned dataset (for reference)

Location: model/train_models.py (works from any folder; paths are resolved
relative to the project root, i.e. the folder that contains app.py).

Run from the project root:   python model/train_models.py
"""
import io
import os
import zipfile
import urllib.request

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

SEED = 2026
TARGET = "diagnosis"          # 1 = malignant, 0 = benign (re-coded below)
# --- paths (resolved from this file's location, so cwd does not matter) ------
_HERE = os.path.dirname(os.path.abspath(__file__))
# This script lives in <project>/model/. If it is run from the project root
# instead, fall back to that folder.
BASE_DIR = os.path.dirname(_HERE) if os.path.basename(_HERE) == "model" else _HERE
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_FILE = os.path.join(BASE_DIR, "test_data.csv")

# Direct download link of the UCI dataset (id = 17). The zip contains wdbc.data.
UCI_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
RAW_FILE = os.path.join(DATA_DIR, "wdbc.data")

# wdbc.data has no header: ID, Diagnosis (M/B), then 30 numeric features made of
# 10 nucleus properties x (mean, standard error, worst).
PROPERTIES = ["radius", "texture", "perimeter", "area", "smoothness",
              "compactness", "concavity", "concave points", "symmetry",
              "fractal dimension"]
FEATURE_NAMES = (
    [f"mean {p}" for p in PROPERTIES]
    + [f"{p} error" for p in PROPERTIES]
    + [f"worst {p}" for p in PROPERTIES]
)


def download_raw_file() -> str:
    """Download the UCI zip, extract wdbc.data and cache it in data/.
    If the download fails (e.g. no internet) fall back to the cached copy."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        print(f"Downloading dataset from UCI: {UCI_URL}")
        with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            with zf.open("wdbc.data") as src, open(RAW_FILE, "wb") as dst:
                dst.write(src.read())
        print(f"Saved raw file to {RAW_FILE}")
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(RAW_FILE):
            print(f"Download failed ({exc}). Using cached copy: {RAW_FILE}")
        else:
            raise RuntimeError(
                "Could not download the dataset from UCI and no cached copy exists. "
                f"Check your internet connection or download {UCI_URL} manually "
                f"and place wdbc.data in the '{DATA_DIR}' folder."
            ) from exc
    return RAW_FILE


def load_data() -> pd.DataFrame:
    path = download_raw_file()
    raw = pd.read_csv(path, header=None,
                      names=["id", "label"] + FEATURE_NAMES)
    df = raw.drop(columns=["id"]).copy()
    # UCI codes the target as M (malignant) / B (benign). Re-code so that the
    # "positive" class (1) is the clinically important one: malignant.
    df[TARGET] = (df["label"] == "M").astype(int)
    df = df.drop(columns=["label"])
    assert df.shape == (569, 31), f"Unexpected dataset shape: {df.shape}"
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
    return df


def build_models() -> dict:
    """Scale-sensitive models get a StandardScaler inside their pipeline,
    so the saved .joblib file is fully self-contained."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3,
                                           random_state=SEED)),
        ]),
        "K-Nearest Neighbor": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale",
                        probability=True, random_state=SEED)),
        ]),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_data()
    df.to_csv(os.path.join(DATA_DIR, "breast_cancer_full.csv"), index=False)
    print(f"Dataset shape: {df.shape}  (features: {df.shape[1] - 1})")
    print(df[TARGET].value_counts().rename({0: "benign", 1: "malignant"}), "\n")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # Held-out test data -> uploaded to the Streamlit app later
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(TEST_FILE, index=False)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        scores = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        fname = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        print(f"{name:<24}", {k: round(v, 4) for k, v in scores.items()})

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    print("\nSaved models, metrics.csv and test_data.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
