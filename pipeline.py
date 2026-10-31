"""Sentiment + aspect analysis, hidden-complaint detection, baseline evaluation.
Usage: python pipeline.py   (reads data/reviews_clean.csv, writes data/reviews_scored.csv)
Optional: add a hand-labelled 0/1 column `gold_hidden` (label ~300 reviews) for honest evaluation."""
import re, numpy as np, pandas as pd, nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import hstack, csr_matrix

ASPECTS = {
    "cleanliness": ["dirty", "clean", "filth", "garbage", "litter", "smell", "hygiene"],
    "crowds": ["crowd", "queue", "busy", "packed", "touts", "rush"],
    "transport": ["taxi", "auto", "bus", "road", "traffic", "train", "cab", "drive"],
    "safety": ["safe", "unsafe", "scam", "theft", "harass", "security"],
    "cost": ["price", "expensive", "overpriced", "cheap", "value", "cost", "overcharge", "fare"],
    "food": ["food", "restaurant", "meal", "cuisine", "seafood", "street food"],
    "scenery": ["view", "scenery", "beach", "fort", "temple", "mountain", "sunset", "garden"],
    "stay": ["hotel", "room", "stay", "hostel", "resort", "host", "staff"],
}
NEG_T = -0.2

def split_sents(t): return [s for s in re.split(r"(?<=[.!?])\s+", t) if s]

def score(df):
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df["compound"] = df.text.map(lambda t: sia.polarity_scores(t)["compound"])
    asp = []
    for t in df.text:
        acc = {}
        for s in split_sents(t):
            c, low = sia.polarity_scores(s)["compound"], s.lower()
            for a, kws in ASPECTS.items():
                if any(k in low for k in kws): acc.setdefault(a, []).append(c)
        asp.append({f"asp_{a}": float(np.mean(acc[a])) if a in acc else np.nan for a in ASPECTS})
    df = pd.concat([df, pd.DataFrame(asp)], axis=1)
    acols = [f"asp_{a}" for a in ASPECTS]
    df["n_complaints"] = (df[acols] < NEG_T).sum(axis=1)
    # weak label: high rating but negative text or aspect complaint
    df["weak_hidden"] = ((df.rating >= 4) & ((df.compound < 0) | (df.n_complaints >= 1))).astype(int)
    return df

ASP_COLS = [f"asp_{a}" for a in ASPECTS]

def features(df, vec=None, rich=False):
    """Text (TF-IDF) + rating. With rich=True also the sentiment signals: VADER score, aspect scores, aspect-mentioned flags."""
    if vec is None: vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(df.text)
    parts = [vec.transform(df.text), csr_matrix(df[["rating"]].values / 5)]
    if rich:
        parts.append(csr_matrix(np.hstack([df[ASP_COLS].fillna(0).values, df[ASP_COLS].notna().values.astype(float),
                                           df[["compound", "n_complaints"]].values, df.text.str.len().values[:, None] / 300])))
    return hstack(parts).tocsr(), vec

def save_results(label, n, scores):
    import json, os                                   # saved so the dashboard can show the results tab
    os.makedirs("data", exist_ok=True)
    json.dump({"target": label, "n_test": int(n), "scores": scores}, open("data/results.json", "w"), indent=1)

def run(df):
    df = score(df)
    hi = df[df.rating >= 4]
    if "gold_hidden" in df and df.gold_hidden.notna().any():
        # Hand labels exist: train on them with 5-fold cross-validation, so every gold review is scored by a model that never saw it.
        gold = hi[hi.gold_hidden.notna()]; y = gold.gold_hidden.astype(int).values
        oof = np.zeros(len(gold), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(gold, y):
            Xtr, vec = features(gold.iloc[tr], rich=True)
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
            oof[te] = m.predict(features(gold.iloc[te], vec, rich=True)[0])
        label, te, preds = "gold_hidden", gold, {
            "Rating only": np.zeros(len(gold), int),                 # a 4-5 star rating never flags anything
            "Sentiment only (VADER<0)": (gold.compound < 0).astype(int).values,
            "TravelMood (LogReg)": oof}
        print(f"Evaluation target: gold_hidden ({len(gold)} hand-labelled reviews, 5-fold cross-validation)")
        Xg, vec = features(gold, rich=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xg, y)   # final model: all gold labels
        rich = True
    else:
        label = "weak_hidden"
        tr, te = train_test_split(hi, test_size=.3, random_state=42, stratify=hi[label])
        Xtr, vec = features(tr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, tr[label])
        preds = {"Rating only": np.zeros(len(te), int), "Sentiment only (VADER<0)": (te.compound < 0).astype(int).values,
                 "TravelMood (LogReg)": clf.predict(features(te, vec)[0])}
        print("Evaluation target: weak_hidden  (rule-based labels, add gold_hidden for a real result)")
        rich = False
    scores = {}
    for k, p in preds.items():
        scores[k] = round(f1_score(te[label].astype(int), p, average="macro", zero_division=0), 3)
        print(f"{k:28s} macro-F1 = {scores[k]:.3f}")
    save_results(label, len(te), scores)
    print(classification_report(te[label].astype(int), preds["TravelMood (LogReg)"], zero_division=0))
    Xall, _ = features(df, vec, rich=rich)
    df["hidden_prob"] = clf.predict_proba(Xall)[:, 1]
    df.loc[df.rating < 4, "hidden_prob"] = 0.0
    return df

if __name__ == "__main__":
    out = run(pd.read_csv("data/reviews_clean.csv"))
    out.to_csv("data/reviews_scored.csv", index=False)
    print("saved data/reviews_scored.csv")
