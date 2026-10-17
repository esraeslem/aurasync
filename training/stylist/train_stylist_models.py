# %% [markdown]
# # AuraSync — Stylist Agent training
# Trains two independent (TF-IDF, LogisticRegression) pairs from ModCloth
# review data: a 3-class fit-prediction model (fit / small / large) and a
# binary sentiment model derived from the `quality` star rating.
#
# Exports four artifacts to aurasync_stylist_outputs/, matching exactly
# what StylistService expects (plain pickle.dump, since the service loads
# with plain pickle.load — no joblib).

# %%
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get("AURASYNC_DATA_DIR", "/Users/mac/Downloads")
MODCLOTH_PATH = os.path.join(DATA_DIR, "modcloth_final_data.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "aurasync_stylist_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# %%
# ---- Load ----
df = pd.read_json(MODCLOTH_PATH, lines=True)
print("raw shape:", df.shape)
print(df["fit"].value_counts())
print(df["quality"].value_counts(dropna=False).sort_index())

# %%
# ---- Fit-prediction: TF-IDF(review_text) -> fit label (3-class) ----
fit_df = df[df["review_text"].notna() & df["fit"].notna()].copy()
fit_df["review_text"] = fit_df["review_text"].astype(str)
print("fit_df shape:", fit_df.shape)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    fit_df["review_text"], fit_df["fit"],
    test_size=0.2, random_state=RANDOM_STATE, stratify=fit_df["fit"],
)

fit_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
X_train_f_vec = fit_tfidf.fit_transform(X_train_f)
X_test_f_vec = fit_tfidf.transform(X_test_f)

fit_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
fit_model.fit(X_train_f_vec, y_train_f)

fit_preds = fit_model.predict(X_test_f_vec)
print("\n=== Fit model — classification report ===")
print(classification_report(y_test_f, fit_preds))
print("=== Fit model — confusion matrix (labels:", list(fit_model.classes_), ") ===")
print(confusion_matrix(y_test_f, fit_preds, labels=fit_model.classes_))
print("accuracy:", round(accuracy_score(y_test_f, fit_preds), 4))

# %%
# ---- Sentiment: TF-IDF(review_text) -> positive/negative, derived from quality ----
# quality>=4 -> positive, quality<=2 -> negative; quality==3 dropped as ambiguous
# rather than forced into either class.
sent_df = df[df["review_text"].notna() & df["quality"].notna()].copy()
sent_df = sent_df[sent_df["quality"] != 3]
sent_df["sentiment"] = np.where(sent_df["quality"] >= 4, "positive", "negative")
sent_df["review_text"] = sent_df["review_text"].astype(str)
print("sent_df shape:", sent_df.shape)
print(sent_df["sentiment"].value_counts())

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    sent_df["review_text"], sent_df["sentiment"],
    test_size=0.2, random_state=RANDOM_STATE, stratify=sent_df["sentiment"],
)

sentiment_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
X_train_s_vec = sentiment_tfidf.fit_transform(X_train_s)
X_test_s_vec = sentiment_tfidf.transform(X_test_s)

sentiment_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
sentiment_model.fit(X_train_s_vec, y_train_s)

sentiment_preds = sentiment_model.predict(X_test_s_vec)
print("\n=== Sentiment model — classification report ===")
print(classification_report(y_test_s, sentiment_preds))
print("=== Sentiment model — confusion matrix (labels:", list(sentiment_model.classes_), ") ===")
print(confusion_matrix(y_test_s, sentiment_preds, labels=sentiment_model.classes_))
print("accuracy:", round(accuracy_score(y_test_s, sentiment_preds), 4))

# %%
# ---- Export — plain pickle.dump to match StylistService's plain pickle.load ----
with open(os.path.join(OUTPUT_DIR, "fit_tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(fit_tfidf, f)
with open(os.path.join(OUTPUT_DIR, "fit_logreg_model.pkl"), "wb") as f:
    pickle.dump(fit_model, f)
with open(os.path.join(OUTPUT_DIR, "sentiment_tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(sentiment_tfidf, f)
with open(os.path.join(OUTPUT_DIR, "sentiment_logreg_model.pkl"), "wb") as f:
    pickle.dump(sentiment_model, f)

print("\nSaved all four artifacts to:", OUTPUT_DIR)
