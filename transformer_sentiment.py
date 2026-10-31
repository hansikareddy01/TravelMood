"""Transformer sentiment (DistilBERT, SST-2) compared with VADER.
Usage:  py transformer_sentiment.py [reviews_per_destination]     (default 100)
Needs:  pip install -r requirements-transformer.txt   (downloads a ~270 MB model the first time)
Writes: data/transformer_scores.csv and adds the transformer baseline to data/results.json"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
df = pd.read_csv("data/reviews_scored.csv")
gold = list(df.index[df.gold_hidden.notna()]) if "gold_hidden" in df else []
rng = np.random.default_rng(0)
pick = set(gold)
for _, g in df.groupby("destination"):
    pick |= set(g.index[rng.permutation(len(g))[:N]])
rows = sorted(pick)
print(f"Scoring {len(rows)} reviews ({len(gold)} gold + up to {N} per destination) with {MODEL}")

from transformers import pipeline
clf = pipeline("sentiment-analysis", model=MODEL, truncation=True, max_length=256)
tf = []
for i in range(0, len(rows), 500):
    out = clf(df.loc[rows[i:i + 500], "text"].tolist(), batch_size=32)
    tf += [o["score"] if o["label"] == "POSITIVE" else -o["score"] for o in out]   # -1 (negative) .. +1 (positive)
    print(f"  {min(i + 500, len(rows))}/{len(rows)}")
pd.DataFrame({"row_id": rows, "tf_sent": tf}).to_csv("data/transformer_scores.csv", index=False)

if gold:
    t = pd.read_csv("data/transformer_scores.csv").set_index("row_id").loc[gold, "tf_sent"]
    g = df.loc[gold]; y = g.gold_hidden.astype(int)
    v = f1_score(y, (g.compound < 0).astype(int), average="macro", zero_division=0)
    x = f1_score(y, (t.values < 0).astype(int), average="macro", zero_division=0)
    print(f"On {len(gold)} gold reviews: VADER-only macro-F1 = {v:.3f} | Transformer-only macro-F1 = {x:.3f}")
    p = Path("data/results.json")
    r = json.load(open(p)) if p.exists() else {"target": "gold_hidden", "scores": {}}
    r["scores"]["Transformer only (DistilBERT)"] = round(x, 3)
    json.dump(r, open(p, "w"), indent=1)
else:
    print("No gold labels found (run label_tool.py merge and pipeline.py first) - skipped the evaluation.")
print("Done. Restart the dashboard to see the transformer comparison.")
