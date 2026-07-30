# ShopSmart

Student ML project: predict whether an online shopping session will generate **revenue**, using a **Decision Tree** classifier on session behaviour features.

**Deploy this app (Streamlit Cloud):**  
https://share.streamlit.io/deploy?repository=Sohaib-Abdullah/descision-tree&branch=main&mainModule=app.py

> Note: Streamlit apps do **not** run on Vercel. Use the link above (Streamlit Community Cloud).

## What's inside

| File | Purpose |
|------|---------|
| `shop_smart_ecommerce.csv` | Session dataset (~12k rows) |
| `shop_smart.ipynb` | Notebook — EDA / train / tune / evaluate |
| `train_model.py` | Trains the tuned pipeline and saves artifacts |
| `app.py` | Streamlit UI (overview, charts, metrics, live predict) |
| `model/` | Saved pipeline + metrics JSON |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

App opens at `http://localhost:8501`.

## Model notes

- Preprocessing: `StandardScaler` (numeric) + `OneHotEncoder` (Month, VisitorType)
- Classifier: `DecisionTreeClassifier(class_weight="balanced")`
- Tuning: `GridSearchCV` on `max_depth` / `min_samples_leaf`, scoring = F1
- Metric focus: **F1 / recall** because purchases are the minority class (~15%)

## Deploy

### Why not Vercel?

**Vercel does not run Streamlit.** Vercel is built for Node/static frontends and short serverless functions. Streamlit needs a long-running Python process with WebSockets, so this app cannot be hosted there.

Use **Streamlit Community Cloud** instead (free, built for this stack).

### Streamlit Community Cloud

1. Open the pre-filled deploy link:  
   https://share.streamlit.io/deploy?repository=Sohaib-Abdullah/descision-tree&branch=main&mainModule=app.py
2. Sign in with the **Sohaib-Abdullah** GitHub account and authorize Streamlit
3. Confirm:
   - Repository: `Sohaib-Abdullah/descision-tree`
   - Branch: `main`
   - Main file: `app.py`
4. Click **Deploy**

Public URL will look like: `https://….streamlit.app`

The trained model under `model/` is already in the repo, so the cloud app starts without running `train_model.py` again.
## Project structure

```
decision-tree/
├── app.py
├── train_model.py
├── shop_smart.ipynb
├── shop_smart_ecommerce.csv
├── requirements.txt
├── model/
│   ├── shopsmart_pipeline.joblib
│   └── metrics.json
└── .streamlit/config.toml
```
