"""
ShopSmart — train Decision Tree pipeline and save artifacts.
Run once before launching the Streamlit app:
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "shop_smart_ecommerce.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "shopsmart_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df["Revenue"] = df["Revenue"].astype(int)
    if df["Weekend"].dtype == object:
        df["Weekend"] = df["Weekend"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0}).astype(int)
    else:
        df["Weekend"] = df["Weekend"].astype(int)

    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]
    return X, y


def build_pipeline(num_features, cat_features) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(num_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_features)),
        ]
    )
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    num_features = X.select_dtypes(include=["int64", "float64", "bool", "int32"]).columns
    cat_features = X.select_dtypes(include=["object", "category", "string"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(num_features, cat_features)

    # baseline fit (same as notebook first pass)
    pipe.fit(X_train, y_train)
    baseline_pred = pipe.predict(X_test)
    baseline_f1 = float(f1_score(y_test, baseline_pred))

    # grid search for best params.
    # shallow trees (depth 4) collapse to "PageValues only" and ignore the rest of the
    # session, so the search range starts deeper to keep the other features in play.
    param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [6, 8, 10, 12],
        "model__min_samples_leaf": [5, 10, 20, 30],
        "model__min_samples_split": [2, 10, 20],
    }
    grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    tree = best.named_steps["model"]
    encoded_names = list(best.named_steps["preprocess"].get_feature_names_out())
    importances = tree.feature_importances_

    # roll one-hot columns back into their original column name for reporting.
    # only categorical columns get prefix-matched, otherwise names like
    # ProductRelated_Duration would be folded into ProductRelated.
    grouped: dict[str, float] = {col: 0.0 for col in X.columns}
    for name, value in zip(encoded_names, importances):
        block, stripped = name.split("__", 1)
        if block == "cat":
            source = next(
                (col for col in cat_features if stripped.startswith(f"{col}_")), stripped
            )
        else:
            source = stripped
        grouped[source] = grouped.get(source, 0.0) + float(value)
    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)

    metrics = {
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "baseline_f1": baseline_f1,
        "cv_best_f1": float(grid.best_score_),
        "best_params": grid.best_params_,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_columns": list(X.columns),
        "num_features": list(num_features),
        "cat_features": list(cat_features),
        "month_values": sorted(X["Month"].unique().tolist()),
        "visitor_types": sorted(X["VisitorType"].unique().tolist()),
        "avg_purchase_prob_test": float(np.mean(y_proba)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "tree_depth": int(tree.get_depth()),
        "tree_leaves": int(tree.get_n_leaves()),
        "feature_importances": {k: round(v, 5) for k, v in ranked},
    }

    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Saved model ->", MODEL_PATH)
    print("Saved metrics ->", METRICS_PATH)
    print("Best params:", grid.best_params_)
    print(f"CV F1: {grid.best_score_:.4f} | Test F1: {metrics['test_f1']:.4f}")
    print(f"ROC AUC: {metrics['test_roc_auc']:.4f} | leaves: {metrics['tree_leaves']}")
    print("Top features:", [k for k, _ in ranked[:6]])


if __name__ == "__main__":
    main()
