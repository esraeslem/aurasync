from __future__ import annotations

from ._io import safe_pickle


class StylistService:
    def __init__(self, fit_tfidf, fit_model, sentiment_tfidf, sentiment_model):
        self.fit_tfidf = safe_pickle(fit_tfidf)
        self.fit_model = safe_pickle(fit_model)
        self.sent_tfidf = safe_pickle(sentiment_tfidf)
        self.sent_model = safe_pickle(sentiment_model)
        self.load_status = {
            "fit_tfidf": self.fit_tfidf is not None, "fit_model": self.fit_model is not None,
            "sentiment_tfidf": self.sent_tfidf is not None, "sentiment_model": self.sent_model is not None,
        }

    @staticmethod
    def _predict(vectorizer, model, text):
        if vectorizer is None or model is None:
            raise RuntimeError("Stylist model artifacts are not available.")
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        probs = None
        if hasattr(model, "predict_proba"):
            probs = {str(c): round(float(p), 4) for c, p in zip(model.classes_, model.predict_proba(X)[0])}
        return str(pred), probs

    def predict_fit(self, text):
        pred, probs = self._predict(self.fit_tfidf, self.fit_model, text)
        return {"prediction": pred, "probabilities": probs}

    def predict_sentiment(self, text):
        pred, probs = self._predict(self.sent_tfidf, self.sent_model, text)
        return {"prediction": pred, "probabilities": probs}
