"""
ShopSmart — Streamlit UI
Purchase-intent prediction using a Decision Tree on session behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "shop_smart_ecommerce.csv"
MODEL_PATH = ROOT / "model" / "shopsmart_pipeline.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"

# palette — quiet academic look, not the usual AI purple theme
NAVY = "#1e3a5f"
INK = "#243447"
MUTED = "#5c6b7a"
LINE = "#c5ced6"
PAPER = "#f7f5f1"
CARD = "#ffffff"
ACCENT = "#2f6f5e"
WARN = "#a65d3b"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {{
            font-family: 'IBM Plex Sans', sans-serif;
            color: {INK};
        }}

        .stApp {{
            background:
                radial-gradient(ellipse 80% 50% at 10% -10%, #e8eef4 0%, transparent 55%),
                linear-gradient(180deg, {PAPER} 0%, #efece6 100%);
        }}

        /* hide streamlit chrome a bit */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1080px;
        }}

        .ss-topbar {{
            border-bottom: 1px solid {LINE};
            padding-bottom: 0.85rem;
            margin-bottom: 1.4rem;
        }}
        .ss-kicker {{
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.35rem;
        }}
        .ss-title {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 2.15rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.15;
            margin: 0;
        }}
        .ss-sub {{
            margin-top: 0.45rem;
            color: {MUTED};
            font-size: 0.98rem;
            max-width: 42rem;
            line-height: 1.45;
        }}

        .ss-panel {{
            background: {CARD};
            border: 1px solid {LINE};
            border-radius: 4px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
        }}
        .ss-panel h3 {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.05rem;
            color: {NAVY};
            margin: 0 0 0.55rem 0;
            font-weight: 600;
        }}
        .ss-muted {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.45; }}

        .ss-metric-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
            margin: 0.8rem 0 1.1rem 0;
        }}
        .ss-metric {{
            background: {CARD};
            border: 1px solid {LINE};
            border-left: 3px solid {NAVY};
            padding: 0.7rem 0.8rem;
        }}
        .ss-metric .label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {MUTED};
        }}
        .ss-metric .value {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.45rem;
            color: {NAVY};
            font-weight: 700;
            margin-top: 0.15rem;
        }}

        .ss-result {{
            border: 1px solid {LINE};
            background: {CARD};
            padding: 1.1rem 1.2rem;
            border-left: 4px solid {ACCENT};
        }}
        .ss-result.warn {{ border-left-color: {WARN}; }}
        .ss-result .verdict {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.35rem;
            color: {NAVY};
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .ss-footer {{
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid {LINE};
            color: {MUTED};
            font-size: 0.82rem;
        }}

        /* streamlit widgets */
        div[data-testid="stSidebar"] {{
            background: #f0ebe3;
            border-right: 1px solid {LINE};
        }}
        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3 {{
            font-family: 'Source Serif 4', Georgia, serif;
            color: {NAVY};
        }}

        @media (max-width: 900px) {{
            .ss-metric-row {{ grid-template-columns: repeat(2, 1fr); }}
            .ss-title {{ font-size: 1.7rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Revenue"] = df["Revenue"].astype(int)
    if df["Weekend"].dtype == object:
        df["Weekend"] = df["Weekend"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0}).astype(int)
    else:
        df["Weekend"] = df["Weekend"].astype(int)
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    return model, metrics


def header() -> None:
    st.markdown(
        """
        <div class="ss-topbar">
          <div class="ss-kicker">ML course project · Decision Tree</div>
          <h1 class="ss-title">ShopSmart</h1>
          <p class="ss-sub">
            Predict whether an online shopping session will end in a purchase,
            using session behaviour features and a tuned Decision Tree classifier.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(metrics: dict) -> None:
    st.markdown(
        f"""
        <div class="ss-metric-row">
          <div class="ss-metric">
            <div class="label">Test F1</div>
            <div class="value">{metrics['test_f1']:.3f}</div>
          </div>
          <div class="ss-metric">
            <div class="label">Accuracy</div>
            <div class="value">{metrics['test_accuracy']:.3f}</div>
          </div>
          <div class="ss-metric">
            <div class="label">Recall (buy)</div>
            <div class="value">{metrics['test_recall']:.3f}</div>
          </div>
          <div class="ss-metric">
            <div class="label">Sessions</div>
            <div class="value">{metrics['n_samples']:,}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(df: pd.DataFrame, metrics: dict) -> None:
    metric_strip(metrics)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown('<div class="ss-panel"><h3>What this project does</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            Online stores get thousands of sessions every day. Only a small share
            actually buy something. ShopSmart takes the usual session signals —
            pages viewed, time spent, bounce/exit rates, month, visitor type —
            and estimates the chance that the visit turns into revenue.

            The model is a **Decision Tree** (not a black-box neural net), so the
            rules stay readable for a coursework report.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ss-panel"><h3>Dataset notes</h3>', unsafe_allow_html=True)
        buy_rate = 100 * df["Revenue"].mean()
        st.markdown(
            f"""
            - Source file: `shop_smart_ecommerce.csv`  
            - Rows: **{len(df):,}** sessions · Features: **{df.shape[1] - 1}**  
            - Purchase rate: **{buy_rate:.1f}%** (class imbalance is real)  
            - Train / test split: 80 / 20, stratified  
            - Best params from GridSearchCV: `max_depth={metrics['best_params']['model__max_depth']}`,
              `min_samples_leaf={metrics['best_params']['model__min_samples_leaf']}`
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        rates = (
            df["Revenue"]
            .value_counts()
            .rename(index={0: "No purchase", 1: "Purchase"})
            .reset_index()
        )
        rates.columns = ["Outcome", "Count"]
        fig = px.bar(
            rates,
            x="Outcome",
            y="Count",
            text="Count",
            color="Outcome",
            color_discrete_map={"No purchase": "#8fa3b5", "Purchase": ACCENT},
        )
        fig.update_traces(textposition="outside", showlegend=False)
        fig.update_layout(
            title="Class balance in the dataset",
            title_font=dict(size=14, color=NAVY, family="Source Serif 4"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=45, b=20, l=10, r=10),
            height=320,
            yaxis_title="",
            xaxis_title="",
            font=dict(color=INK, family="IBM Plex Sans"),
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Purchase sessions are the minority — that is why F1 / recall matter "
            "more than raw accuracy for this task."
        )


def page_explore(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="ss-panel"><h3>Quick look at the data</h3>'
        '<p class="ss-muted">A few plots I used while checking feature behaviour before training.</p></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_rev = (
            df.groupby("Month")["Revenue"]
            .mean()
            .reindex([m for m in month_order if m in df["Month"].unique()])
            .reset_index()
        )
        month_rev["Purchase rate"] = month_rev["Revenue"] * 100
        fig = px.bar(
            month_rev,
            x="Month",
            y="Purchase rate",
            color_discrete_sequence=[NAVY],
        )
        fig.update_layout(
            title="Purchase rate by month",
            title_font=dict(size=14, color=NAVY, family="Source Serif 4"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(t=45, b=20),
            yaxis_title="%",
            xaxis_title="",
            font=dict(color=INK),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        vt = (
            df.groupby("VisitorType")["Revenue"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "rate", "count": "n"})
        )
        vt["Purchase rate"] = vt["rate"] * 100
        fig = px.bar(
            vt,
            x="VisitorType",
            y="Purchase rate",
            text="n",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_layout(
            title="Purchase rate by visitor type",
            title_font=dict(size=14, color=NAVY, family="Source Serif 4"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(t=45, b=20),
            yaxis_title="%",
            xaxis_title="",
            font=dict(color=INK),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### PageValues vs purchase")
    sample = df.sample(n=min(2500, len(df)), random_state=7).copy()
    sample["Outcome"] = sample["Revenue"].map({0: "No purchase", 1: "Purchase"})
    fig = px.scatter(
        sample,
        x="PageValues",
        y="ExitRates",
        color="Outcome",
        opacity=0.55,
        color_discrete_map={"No purchase": "#9aa8b5", "Purchase": WARN},
    )
    fig.update_layout(
        title="Higher PageValues usually sit with purchases (sample of sessions)",
        title_font=dict(size=14, color=NAVY, family="Source Serif 4"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(t=45, b=20),
        font=dict(color=INK),
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Raw column summary"):
        st.dataframe(df.describe(include="all").T, width="stretch")


def page_model(metrics: dict) -> None:
    metric_strip(metrics)

    st.markdown(
        f"""
        <div class="ss-panel">
          <h3>Training setup</h3>
          <p class="ss-muted">
            Pipeline: StandardScaler on numeric columns + OneHotEncoder on
            Month / VisitorType, then DecisionTreeClassifier with
            <code>class_weight='balanced'</code>. Hyperparameters chosen with
            5-fold GridSearchCV on F1.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.markdown("**Best hyperparameters**")
        st.json(metrics["best_params"])
        st.markdown(
            f"""
            | Metric | Value |
            |---|---|
            | CV best F1 | {metrics['cv_best_f1']:.4f} |
            | Baseline F1 (before tune) | {metrics['baseline_f1']:.4f} |
            | Test precision | {metrics['test_precision']:.4f} |
            | Test recall | {metrics['test_recall']:.4f} |
            | Test F1 | {metrics['test_f1']:.4f} |
            """
        )

    with c2:
        cm = np.array(metrics["confusion_matrix"])
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred: No", "Pred: Yes"],
                y=["Actual: No", "Actual: Yes"],
                colorscale=[[0, "#eef2f5"], [1, NAVY]],
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        fig.update_layout(
            title="Confusion matrix (test set)",
            title_font=dict(size=14, color=NAVY, family="Source Serif 4"),
            height=360,
            margin=dict(t=45, b=20, l=80, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK),
        )
        st.plotly_chart(fig, width="stretch")

    st.info(
        "I optimised for F1 because purchases are rare. Accuracy alone looks high "
        "even if the model mostly predicts 'no purchase'."
    )

    importances = metrics.get("feature_importances", {})
    if importances:
        imp = (
            pd.DataFrame({"Feature": list(importances), "Importance": list(importances.values())})
            .query("Importance > 0")
            .sort_values("Importance")
        )
        imp["Feature"] = imp["Feature"].map(lambda c: PRETTY_NAMES.get(c, c))
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
        fig.update_traces(marker_color=NAVY)
        fig.update_layout(
            title="Which features the tree actually splits on",
            title_font=dict(size=14, color=NAVY, family="Source Serif 4"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
            margin=dict(t=45, b=20, l=10, r=10),
            yaxis_title="",
            font=dict(color=INK),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Tree depth {metrics.get('tree_depth', '?')} with "
            f"{metrics.get('tree_leaves', '?')} leaves, ROC AUC "
            f"{metrics.get('test_roc_auc', float('nan')):.3f}. A shallower tree scored a similar F1 "
            "but leaned almost entirely on page value, which made the app feel unresponsive, "
            "so the search range was widened."
        )


PRETTY_NAMES = {
    "PageValues": "Page value score",
    "Month": "Month",
    "ProductRelated_Duration": "Time on product pages",
    "Administrative_Duration": "Time on account pages",
    "ExitRates": "Exit rate",
    "BounceRates": "Bounce rate",
    "ProductRelated": "Product pages viewed",
    "Informational_Duration": "Time on info pages",
    "TrafficType": "Traffic source",
    "Administrative": "Account pages viewed",
    "Informational": "Info pages viewed",
    "Region": "Region",
    "Weekend": "Weekend visit",
    "OperatingSystems": "Operating system",
    "Browser": "Browser",
    "VisitorType": "Visitor type",
    "SpecialDay": "Special day proximity",
}


def _fmt(x: float) -> str:
    return f"{x:,.4f}" if abs(x) < 1 else f"{x:,.2f}"


def explain_path(model, row: pd.DataFrame) -> list[str]:
    """Human-readable list of the splits this session actually went through."""
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    tree = clf.tree_

    encoded = pre.transform(row)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    names = list(pre.get_feature_names_out())

    scaler = pre.named_transformers_["num"]
    num_cols = list(pre.transformers_[0][2])

    indicator = clf.decision_path(encoded)
    nodes = list(indicator.indices[indicator.indptr[0] : indicator.indptr[1]])

    steps: list[str] = []
    for position, node in enumerate(nodes):
        feature_idx = tree.feature[node]
        if feature_idx < 0:  # leaf
            continue
        # direction comes from the path itself, not a float comparison
        went_left = tree.children_left[node] == nodes[position + 1]

        block, stripped = names[feature_idx].split("__", 1)
        threshold = float(tree.threshold[node])
        value = float(encoded[0, feature_idx])

        if block == "num":
            pos = num_cols.index(stripped)
            threshold = threshold * scaler.scale_[pos] + scaler.mean_[pos]
            value = value * scaler.scale_[pos] + scaler.mean_[pos]
            label = PRETTY_NAMES.get(stripped, stripped)
            sign = "\u2264" if went_left else ">"
            steps.append(f"{label} {sign} {_fmt(threshold)}  (this session: {_fmt(value)})")
        else:
            column = next((c for c in ("Month", "VisitorType") if stripped.startswith(f"{c}_")), stripped)
            category = stripped[len(column) + 1 :] if stripped.startswith(f"{column}_") else stripped
            label = PRETTY_NAMES.get(column, column)
            state = "is not" if went_left else "is"
            steps.append(f"{label} {state} \u201c{category}\u201d")
    return steps


def default_inputs(df: pd.DataFrame) -> dict:
    # medians / modes — sensible starting point for the form
    return {
        "Administrative": int(df["Administrative"].median()),
        "Administrative_Duration": float(df["Administrative_Duration"].median()),
        "Informational": int(df["Informational"].median()),
        "Informational_Duration": float(df["Informational_Duration"].median()),
        "ProductRelated": int(df["ProductRelated"].median()),
        "ProductRelated_Duration": float(df["ProductRelated_Duration"].median()),
        "BounceRates": float(df["BounceRates"].median()),
        "ExitRates": float(df["ExitRates"].median()),
        "PageValues": float(df["PageValues"].median()),
        "SpecialDay": float(df["SpecialDay"].median()),
        "Month": "Nov",
        "OperatingSystems": int(df["OperatingSystems"].mode().iloc[0]),
        "Browser": int(df["Browser"].mode().iloc[0]),
        "Region": int(df["Region"].mode().iloc[0]),
        "TrafficType": int(df["TrafficType"].mode().iloc[0]),
        "VisitorType": "Returning_Visitor",
        "Weekend": 0,
    }


def seed_session_state(df: pd.DataFrame, metrics: dict) -> None:
    if "in_PageValues" in st.session_state:
        return
    for key, value in default_inputs(df).items():
        st.session_state[f"in_{key}"] = value


def load_real_session(df: pd.DataFrame, want_purchase: bool | None = None) -> None:
    pool = df if want_purchase is None else df[df["Revenue"] == int(want_purchase)]
    sample = pool.sample(1).iloc[0]
    for col in df.columns:
        if col == "Revenue":
            continue
        value = sample[col]
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        st.session_state[f"in_{col}"] = value
    st.session_state["actual_revenue"] = int(sample["Revenue"])


def page_predict(df: pd.DataFrame, model, metrics: dict) -> None:
    st.markdown(
        """
        <div class="ss-panel">
          <h3>Try a session</h3>
          <p class="ss-muted">
            The prediction updates live as you edit the sidebar — no need to press
            anything. Fields are ordered by how much the trained tree relies on them,
            and the rule trace below shows exactly which splits your session passed
            through.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    seed_session_state(df, metrics)
    months = metrics.get("month_values", sorted(df["Month"].unique().tolist()))
    visitors = metrics.get("visitor_types", sorted(df["VisitorType"].unique().tolist()))

    with st.sidebar:
        st.markdown("### Session inputs")
        st.caption("Ordered by feature importance.")

        st.number_input(
            "Page value score",
            0.0,
            400.0,
            step=1.0,
            key="in_PageValues",
            help="Average value of the pages viewed. Strongest signal in the tree (~65%).",
        )
        st.selectbox("Month", months, key="in_Month", help="Second strongest signal (~16%).")
        st.number_input(
            "Time on product pages (sec)",
            0.0,
            70000.0,
            step=10.0,
            key="in_ProductRelated_Duration",
        )
        st.number_input("Product pages viewed", 0, 700, key="in_ProductRelated")
        st.slider("Exit rate", 0.0, 0.2, step=0.001, key="in_ExitRates")
        st.slider("Bounce rate", 0.0, 0.2, step=0.001, key="in_BounceRates")

        with st.expander("More session details"):
            st.number_input(
                "Time on account pages (sec)", 0.0, 4000.0, step=1.0, key="in_Administrative_Duration"
            )
            st.number_input("Account pages viewed", 0, 30, key="in_Administrative")
            st.number_input(
                "Time on info pages (sec)", 0.0, 3000.0, step=1.0, key="in_Informational_Duration"
            )
            st.number_input("Info pages viewed", 0, 30, key="in_Informational")
            st.number_input("Traffic source code", 1, 20, key="in_TrafficType")
            st.number_input("Region", 1, 9, key="in_Region")
            st.number_input("Operating system code", 1, 8, key="in_OperatingSystems")
            st.number_input("Browser code", 1, 13, key="in_Browser")
            st.selectbox("Visitor type", visitors, key="in_VisitorType")
            st.slider("Special day proximity", 0.0, 1.0, step=0.2, key="in_SpecialDay")
            st.selectbox("Weekend visit", [0, 1], key="in_Weekend", format_func=lambda v: "Yes" if v else "No")

        st.divider()
        st.caption("Load a real row from the CSV to sanity-check the model:")
        c1, c2 = st.columns(2)
        c1.button(
            "Buyer session",
            width="stretch",
            on_click=load_real_session,
            args=(df, True),
        )
        c2.button(
            "Non-buyer session",
            width="stretch",
            on_click=load_real_session,
            args=(df, False),
        )

    row = pd.DataFrame([{col: st.session_state[f"in_{col}"] for col in metrics["feature_columns"]}])
    row = row[metrics["feature_columns"]]

    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1])
    label = "Likely to purchase" if pred == 1 else "Unlikely to purchase"
    css_class = "ss-result" if pred == 1 else "ss-result warn"

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(
            f"""
            <div class="{css_class}">
              <div class="verdict">{label}</div>
              <div>Estimated purchase probability: <b>{proba * 100:.1f}%</b></div>
              <div style="margin-top:0.35rem;color:{MUTED};font-size:0.9rem;">
                Class label uses the standard 0.5 cut-off.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        actual = st.session_state.get("actual_revenue")
        if actual is not None:
            outcome = "did purchase" if actual == 1 else "did not purchase"
            agreement = "matches" if actual == pred else "misses"
            st.caption(
                f"Loaded CSV session — the real visitor {outcome}, so this prediction {agreement}."
            )

        st.markdown("#### Rules this session followed")
        steps = explain_path(model, row)
        if steps:
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1)))
        else:
            st.caption("Tree returned a root leaf — no splits to show.")

    with right:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 30, "color": NAVY}},
                title={"text": "Purchase probability", "font": {"size": 14, "color": MUTED}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": ACCENT if pred == 1 else WARN},
                    "bgcolor": "#eef1f4",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "#e4e8ec"},
                        {"range": [35, 65], "color": "#d5dde4"},
                        {"range": [65, 100], "color": "#c5d0da"},
                    ],
                    "threshold": {
                        "line": {"color": NAVY, "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(t=45, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Row sent to the pipeline"):
            shown = row.T.rename(columns={0: "value"}).astype(str)
            st.dataframe(shown, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="ShopSmart · Decision Tree",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    header()

    model, metrics = load_model()
    if model is None:
        st.error(
            "Model files missing. From the project folder run:\n\n"
            "`python train_model.py`\n\nThen refresh this page."
        )
        st.stop()

    df = load_dataframe()

    tabs = st.tabs(["Overview", "Explore data", "Model results", "Predict"])
    with tabs[0]:
        page_overview(df, metrics)
    with tabs[1]:
        page_explore(df)
    with tabs[2]:
        page_model(metrics)
    with tabs[3]:
        page_predict(df, model, metrics)

    st.markdown(
        """
        <div class="ss-footer">
          ShopSmart — decision-tree purchase prediction · dataset: online shopper sessions ·
          built as a student ML project (sklearn + Streamlit)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
