# AuraSync: Multi-Agent Wardrobe & Styling System

Capstone project for the **Miuul AI & Data Science Bootcamp (21st Cohort)**.

AuraSync is a FastAPI backend with four specialised "agents", each built on a classical ML / analytics pipeline. A web app sits on top, along with an LLM chatbot (the "AI Concierge") that calls the agents as tools. All models are trained and evaluated on public retail and fashion-review datasets.

> **Status:** working local prototype. Not deployed. No mobile app. See [Limitations](#limitations).

---

## Repository layout

```
main.py                     FastAPI entrypoint (+ CORS, /health)
app/
  config.py                 artifact paths (AURASYNC_DATA_DIR), LLM provider settings
  routers/                  economist, retail_strategist, stylist, visionary, chat, frontend (/app)
  services/                 one service per agent + catalog, registry, concierge (LLM + tools)
frontend/                   wardrobe web app (served at /app)
scripts/test_chat.py        end-to-end smoke test for the chatbot
training/                   scripts that produced the model artifacts
  economist/                churn/fatigue model + churn feature-store export
  retail_strategist/        RFM/CLTV script, ARL notebook (fpgrowth)
  stylist/                  fit & sentiment classifiers
  visionary/                colour extraction, FashionSigLIP embeddings, diagnostics
  eda/                      EDA + data-quality checks
extras/email_service.py     mock email sender (logs only, cut from final scope)
Start AuraSync.command      macOS double-click launcher
```

---

## Datasets (not included)

| Dataset | Used for |
|---|---|
| [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (~105,542 articles, ~31.8M transactions) | RFM, CLTV, ARL, visual search, style personas, churn |
| Mercari (filtered to apparel) | Resale price model |
| ModCloth (`modcloth_final_data.json`) | Fit prediction & review sentiment |

Datasets and trained artifacts (`.csv`, `.npy`, `.pkl`) are git-ignored. Put them in `AURASYNC_DATA_DIR` (default `~/Downloads`). `/health` reports which artifacts loaded.

---

## The four agents

### 1. Retail Strategist: segmentation & "complete the look"
- **RFM** segmentation and **CLTV** scoring → `customer_features.csv`
- **Association Rule Learning** with `mlxtend` `fpgrowth` on `product_type_name` → `arl_rules.csv`
- Recommendation path: `article_id → product_type → ARL rules → recommended types → real catalogue articles`
- Rules are category-level, not article-level.

### 2. Visionary: computer vision
- Dominant **colour extraction** → `color_features.csv`
- **Visual search** with **Marqo-FashionSigLIP** (`open_clip`): 105,100 × 768 embeddings, cosine similarity
- **Style persona clustering** (K = 7)

Findings:
- Vanilla CLIP showed strong anisotropy (mean top-1 similarity 0.956) and poor retrieval. Marqo-SigLIP reduced this to 0.926 and fixed retrieval.
- Compared with vanilla CLIP, SigLIP gained ~+14.7 pts product-type precision@5 but lost ~5 pts colour-group precision. The cause of the colour regression is an open question.
- Flat KMeans clustered by garment type, not aesthetic. Hierarchical clustering improved cross-category mixing for ~46% of the catalogue, **but 4 of 7 personas are still dominated by one garment type.**

### 3. Stylist: fit & sentiment (NLP)
TF-IDF + Logistic Regression on ModCloth reviews:

| Model | Accuracy |
|---|---|
| Fit (small / fit / large, balanced class weights) | 74% |
| Sentiment | 90% |

### 4. Economist: style fatigue & resale price
- **Churn / style fatigue**: LightGBM + `RandomizedSearchCV`. The target is "no purchase in the next 56 days". `recency_days` dominates feature importance, and tuning gave little gain.
- Live scoring uses a per-customer feature store (`churn_customer_features.csv`, 1,362,281 customers; mean churn probability 0.27, 21.7% ≥ 0.5) built by `training/economist/export_churn_features.py`.
- **Resale price**: LightGBM on Mercari apparel listings (log price, TF-IDF on name/description). The API maps an H&M article onto the Mercari features, assuming condition 2 ("gently used") and brand "H&M".

---

## AI Concierge (LLM + tools)

`app/services/concierge_service.py` gives the LLM 9 tools: article details, recommendations, customer profile, colour, visual search, style persona, fit/sentiment, churn risk and resale price.

- Runs **locally via Ollama** (`qwen3:4b-instruct`), with no paid API. An OpenAI back-end is optional.
- A keyword router (English and Turkish) runs the obvious tools first, then one LLM call writes the answer.
- Tool errors are passed to the LLM instead of being filled in with guesses, and article IDs that no tool returned are removed from replies.
- Example run (`scripts/test_chat.py`): article 0108775015 → black vest top, est. resale $19.30; "tight around the waist" → fit "small" (0.90), sentiment negative; sample customer → churn probability 0.0.

---

## Web app (`/app`)

Sections: **My Wardrobe**, **Discover**, **Outfits**, **Agent Studio**, **AI Concierge**, **My Profile**.
- Browse a **5,000-article subset** of the H&M catalogue (search, group filters, sort)
- Add pieces to a wardrobe, with counts of product types, colours and groups
- **"Complete your look"** suggestions from article metadata (not generated images)
- Items show colour-coded placeholder icons, not product photos

---

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env               # set AURASYNC_DATA_DIR if data is not in ~/Downloads
ollama pull qwen3:4b-instruct
python3 -m uvicorn main:app --reload
# web app: http://127.0.0.1:8000/app   ·   API docs: http://127.0.0.1:8000/docs

# smoke test (second terminal)
python3 scripts/test_chat.py 0108775015 <customer_id>
```

Developed on a MacBook Air M1 (8 GB), Python 3.9.
Pickled models need the **same scikit-learn version** for training and serving (pinned in `requirements.txt`).

---

## Limitations

- **Local prototype only**: not deployed, no auth, no real users.
- **No Flutter / mobile app**; the front end is a local web app.
- Users pick items from the H&M catalogue; they cannot upload photos of their own clothes.
- **No A/B test was run.**
- The local 4B LLM sometimes gets details wrong in its wording (e.g. it said the resale price came from "H&M and Mercari data", but the model was trained on Mercari only). Tool numbers are correct; the text around them is not guaranteed.
- The resale model was trained on Mercari, so prices for H&M articles are an approximation across two datasets.
- Training scripts contain hard-coded local paths (`/Users/mac/Downloads/...`); edit them before re-running.
- ARL works at category level; style personas only partly capture aesthetic style.
- The planned 5th agent (CRM Outreach Concierge) was **cut** from scope; `extras/email_service.py` is a mock.

---

## Gotchas
- H&M `article_id` loses its leading zero when read as int → always `.astype(str).str.zfill(10)`.
- `arl_rules.csv` stores categories as plain strings, sometimes comma-joined (`"Trousers, Blouse"`).
- Check column names before writing code (e.g. `style_persona` / `style_persona_name`).

## Tech stack
Python · pandas · scikit-learn · LightGBM · mlxtend · lifetimes · open_clip / PyTorch · FastAPI · Uvicorn · Ollama

## Author
Esra Eslem Savaş, BSc Statistics (Data Science & Business Analytics), METU
