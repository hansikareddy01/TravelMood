"""Gold labelling helper.
  python label_tool.py make    -> writes labeling_sheet.csv (300 reviews) - open in Excel, fill gold_hidden with 1 or 0
  python label_tool.py merge   -> reads the filled sheet, adds gold_hidden to data/reviews_clean.csv
Then run: python pipeline.py
Label 1 = a 4-5 star review whose TEXT contains a real complaint (dirty, overpriced, crowded, unsafe, bad service...).
Label 0 = text is genuinely positive/neutral. Praise with a small passing gripe about the place itself counts as 1."""
import sys, pandas as pd
N = 300
if sys.argv[1:] == ["make"]:
    d = pd.read_csv("data/reviews_scored.csv"); d["id"] = d.index
    hi = d[d.rating >= 4]
    flagged = hi[hi.weak_hidden == 1]
    a = flagged.sample(min(N // 2, len(flagged)), random_state=1)
    b = hi.drop(a.index).sample(N - len(a), random_state=1)   # random unflagged: keeps the sample honest
    s = pd.concat([a, b]).sample(frac=1, random_state=2)[["id", "destination", "rating", "text"]]
    s["gold_hidden"] = ""
    s.to_csv("labeling_sheet.csv", index=False, encoding="utf-8-sig")
    print(f"wrote labeling_sheet.csv with {len(s)} rows - fill gold_hidden (1 or 0), save as CSV")
elif sys.argv[1:] == ["merge"]:
    s = pd.read_csv("labeling_sheet.csv")
    s = s[s.gold_hidden.isin([0, 1, "0", "1", 0.0, 1.0])]
    c = pd.read_csv("data/reviews_clean.csv")
    lab = dict(zip(zip(s.destination, s.text), s.gold_hidden.astype(float)))   # match by text, so labels survive a re-sample
    c["gold_hidden"] = [lab.get((d, t), float("nan")) for d, t in zip(c.destination, c.text)]
    c.to_csv("data/reviews_clean.csv", index=False)
    print(f"matched {c.gold_hidden.notna().sum()} of {len(s)} labelled reviews; positives: {int(c.gold_hidden.sum())}")
else:
    print(__doc__)
