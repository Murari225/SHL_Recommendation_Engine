# evaluate_recall.py
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

META = "data/metadata.csv"
EMB = "data/embeddings.npy"
NN = "data/nn_model.joblib"
EXCEL = "Gen_AI Dataset.xlsx"

meta = pd.read_csv(META)
embs = np.load(EMB)
nbrs = joblib.load(NN)
model = SentenceTransformer("all-mpnet-base-v2")

def norm(u):
    if pd.isna(u): return ""
    u = str(u).strip().lower()
    return u[:-1] if u.endswith("/") else u

df = pd.read_excel(EXCEL)
gt = df.groupby("Query")["Assessment_url"].apply(lambda x: set(norm(i) for i in x.dropna())).to_dict()

def rec(q, k=10):
    qv = model.encode([q], convert_to_numpy=True).astype("float32")
    dists, idxs = nbrs.kneighbors(qv, n_neighbors=k)
    urls = [norm(meta.iloc[i]["canonical_url"]) for i in idxs[0]]
    return urls

scores = []
for query, truth in gt.items():
    preds = rec(query, 10)
    hit = any(p in truth for p in preds)
    scores.append(1 if hit else 0)

print("Queries evaluated:", len(scores))
print("Mean Recall@10:", sum(scores)/len(scores) if scores else 0.0)
