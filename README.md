# TravelMood
1. `pip install -r requirements.txt`
2. Data: `python data_prep.py raw.csv` (columns: destination, rating, text, date) or `python data_prep.py --demo`
3. `python pipeline.py`  -> scores reviews, trains the mismatch model, prints macro-F1 vs rating-only and sentiment-only baselines
4. `export ANTHROPIC_API_KEY=...` (optional) then `streamlit run app.py`

Notes for the report
- Hidden complaint = 4-5★ review with negative text or a negative aspect sentence. Label ~300 reviews by hand into `gold_hidden`; without it the evaluation uses rule-derived weak labels and is optimistic.
- Reviews rarely state expectations, so mismatch is a proxy, not a measure of expectation gaps.
- Upgrade path: swap VADER for a pretrained transformer, add BERTopic for topic discovery, replace keyword aspects with a multi-label classifier.
