# api.py (safe, sanitizing version)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import math, traceback, sys, re

META = "data/metadata.csv"
EMB = "data/embeddings.npy"
NN = "data/nn_model.joblib"

app = FastAPI(title="SHL Recommender (No FAISS) - sanitized")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

# Text-match booster: promotes candidates whose canonical_text contains query tokens
def boost_by_text_match(results, query, boost=0.25):
    """
    Boost items whose canonical_text contains query tokens (or multi-word phrases).
    results: list of dicts with keys 'score' and 'canonical_text' (and skills_tags)
    query: user query string
    boost: score to add per match (tuneable)
    """
    if not query:
        return results
    q = query.lower()
    # list of multi-word phrases to detect
    multi_phrases = [
        "power bi", "data warehousing", "machine learning", "deep learning",
        "amazon web services", "amazon aws"
    ]
    desired = set()
    for p in multi_phrases:
        if p in q:
            desired.add(p)
    tokens = [t for t in re.split(r'\W+', q) if len(t) > 1]
    for t in tokens:
        desired.add(t)

    for r in results:
        base = float(r.get("score", 0.0))
        canon = (r.get("canonical_text") or "").lower()
        matches = 0
        for d in desired:
            if d in canon:
                matches += 1
        r["score"] = base + matches * boost
        r["_text_match_count"] = matches
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# Load resources at startup
try:
    df = pd.read_csv(META)
    embs = np.load(EMB)
    nbrs = joblib.load(NN)
    # load model (sentence-transformers)
    model = SentenceTransformer("all-mpnet-base-v2")
    print("API: Loaded metadata, embeddings, NN and model.")
except Exception:
    print("API startup error:", file=sys.stderr)
    traceback.print_exc()
    raise

class Req(BaseModel):
    query: str
    top_k: int = 10

def safe_float(x, default=0.0):
    """Return finite float or default if NaN/inf/None."""
    try:
        if x is None:
            return float(default)
        f = float(x)
        if not math.isfinite(f):
            return float(default)
        return f
    except Exception:
        return float(default)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recommend")
def recommend(req: Req):
    try:
        q = (req.query or "").strip()
        k = int(req.top_k) if req.top_k and int(req.top_k) > 0 else 10

        # encode query
        qv = model.encode([q], convert_to_numpy=True).astype("float32")

        # choose neighbors: at least max(k,10) for safety, but not more than dataset size
        n_neighbors = min(max(k, 10), len(embs))
        dists, idxs = nbrs.kneighbors(qv, n_neighbors=n_neighbors)

        out = []
        for dist, idx in zip(dists[0].tolist(), idxs[0].tolist()):
            # guard index
            if idx is None or idx < 0 or idx >= len(df):
                continue
            row = df.iloc[int(idx)].to_dict()
            score = safe_float(1.0 - dist, default=0.0)

            # sanitize metadata fields
            aid = str(row.get("assessment_id", "")) if row.get("assessment_id") is not None else ""
            aname = str(row.get("assessment_name", "")) if row.get("assessment_name") is not None else ""
            curl = str(row.get("canonical_url", "")) if row.get("canonical_url") is not None else ""
            ttype = str(row.get("test_type", "")) if row.get("test_type") is not None else ""
            skills = str(row.get("skills_tags", "")) if row.get("skills_tags") is not None else ""
            canonical = str(row.get("canonical_text", "")) if row.get("canonical_text") is not None else ""

            # clamp weird scores
            if not math.isfinite(score) or abs(score) > 1e6:
                print(f"[WARN] non-finite score for idx={idx}, raw_dist={dist}, clamping to 0.0")
                score = 0.0

            out.append({
                "assessment_id": aid,
                "assessment_name": aname,
                "canonical_url": curl,
                "test_type": ttype,
                "skills_tags": skills,
                "canonical_text": canonical,   # keep for reranking only
                "score": float(score)
            })

        # Apply text-match boosting (promotes candidates whose page text actually contains query tokens)
        out = boost_by_text_match(out, q, boost=0.25)

        # Prepare final output: remove internal fields we don't want returned to client
        final = []
        for r in out[:k]:
            item = {
                "assessment_id": r.get("assessment_id", ""),
                "assessment_name": r.get("assessment_name", ""),
                "canonical_url": r.get("canonical_url", ""),
                "test_type": r.get("test_type", ""),
                "skills_tags": r.get("skills_tags", ""),
                "score": float(r.get("score", 0.0))
            }
            final.append(item)

        return {"query": q, "recommendations": final}

    except Exception as e:
        # log full traceback to server console and return 500 with message
        print("Exception in /recommend:", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

