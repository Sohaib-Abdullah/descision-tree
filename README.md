# ShopSmart

Student ML project: predict whether an online shopping session will generate **revenue**, using a **Decision Tree** classifier on session behaviour features.

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

### Streamlit Community Cloud (recommended for this app)

Streamlit apps do **not** run on Vercel (Vercel is for Node/static frontends). For this project use [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub
2. Go to share.streamlit.io → **New app**
3. Select the repo, branch `main`, main file `app.py`
4. Deploy

The cloud build will install `requirements.txt`. Make sure `model/` (trained `.joblib` + `metrics.json`) is committed, **or** add a startup step that runs `train_model.py` before the app (slower cold start).

### GitHub

```bash
git init
git add .
git commit -m "Add ShopSmart decision-tree project and Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

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
