"""Adapter for Review_db.csv (City, Place, Review, Rating, Raw_Review...).
Samples N reviews per destination and writes raw_sample.csv in TravelMood format.
Usage: python prep_review_db.py path/to/Review_db.csv [N_per_destination]"""
import sys, pandas as pd
# Goa is split into towns in the dataset, so they are merged into one destination
GROUPS = {"Goa": ["Baga", "Calangute", "Candolim", "Goa Velha", "Old Goa", "Nagoa"], "Kochi": ["Kochi (Cochin)", "Fort Kochi"],
          "Ranthambore": ["Sawai Madhopur"], "Mysuru": ["Mysuru (Mysore)"], "Varkala": ["Varkala Town"],
          "Andaman Islands": ["South Andaman Island"]}
CITIES = [  # big cities (keep these: the gold labels come from them)
    "Jaipur", "Udaipur", "Agra", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "New Delhi", "Pune",
    # North and Himalaya
    "Amritsar", "Shimla", "Manali", "McLeod Ganj", "Rishikesh", "Mussoorie", "Nainital", "Leh", "Srinagar", "Gulmarg", "Varanasi",
    # Rajasthan and central
    "Jodhpur", "Jaisalmer", "Pushkar", "Mount Abu", "Khajuraho",
    # East and northeast
    "Darjeeling", "Gangtok", "Shillong", "Puri",
    # West
    "Ahmedabad", "Lonavala", "Mahabaleshwar",
    # South (Kerala, Tamil Nadu, Karnataka, Puducherry)
    "Munnar", "Alappuzha", "Thekkady", "Kovalam", "Ooty", "Kodaikanal", "Madikeri", "Hampi", "Pondicherry", "Madurai", "Rameswaram", "Kanyakumari"]
RENAME = {t: g for g, ts in GROUPS.items() for t in ts}
src, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1500
keep = set(CITIES) | set(RENAME)
parts = [ch[ch.City.isin(keep)] for ch in pd.read_csv(src, usecols=["City", "Rating", "Raw_Review"], chunksize=200_000, on_bad_lines="skip")]
df = pd.concat(parts)
df["City"] = df.City.replace(RENAME)
df = pd.concat([g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("City")])
out = df.rename(columns={"City": "destination", "Rating": "rating", "Raw_Review": "text"})
out["date"] = ""
out.to_csv("raw_sample.csv", index=False)
print(out.destination.value_counts().to_dict())
