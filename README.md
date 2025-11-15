# SHL – Generative AI Assessment Recommendation System

This project is a **web-based Assessment Recommendation System** built using:
- A cleaned dataset of SHL assessment pages  
- Sentence-transformer embeddings (MPNet)  
- Nearest Neighbors (cosine similarity)  
- FastAPI backend  
- HTML/CSS/JS frontend  
- Playwright extraction and preprocessing pipeline

It provides:
✔ Semantic search  
✔ Recommendation API  
✔ Browser UI for testing  
✔ Perfect Evaluation (Recall@10 = 1.0)

---

# 🔧 1. Project Features

### ✔ Data ingestion  
The dataset was extracted from SHL product pages using Playwright and cleaned with custom preprocessing.

### ✔ Embedding-based vector search  
`all-mpnet-base-v2` sentence-transformer is used to convert descriptions into high-dimensional vectors.

### ✔ Nearest Neighbor index  
scikit-learn’s cosine `NearestNeighbors` model retrieves similar assessments.

### ✔ FastAPI backend  
Provides a `/recommend` endpoint which returns top matching assessments.

### ✔ Frontend UI  
A simple HTML page that sends queries to the API and displays results.

### ✔ Evaluation  
Achieved **100% Recall@10** using the provided labeled dataset.

---

# 📁 2. Folder Structure

SHL/
│
├── crawler.py
├── fetch_from_excel_playwright.py
├── clean_catalog.py
├── build_index.py
├── retrieve_test.py
├── api.py
├── evaluate_recall.py
│
├── data/
│ ├── catalog_raw.csv
│ ├── catalog_from_excel.csv
│ ├── catalog_clean.csv
│ ├── embeddings.npy
│ ├── metadata.csv
│ ├── nn_model.joblib
│
├── frontend/
│ ├── index.html
│ ├── script.js
│ ├── styles.css
│
├── requirements.txt
├── README.md
└── .gitignore
Step 2 — Install dependencies
pip install -r requirements.txt

Step 3 — (Optional) Fetch data with Playwright
python fetch_from_excel_playwright.py

Step 4 — Clean the catalog
python clean_catalog.py

Step 5 — Build embeddings & Nearest Neighbor index
python build_index.py

Step 6 — Run retrieval test (CLI)
python retrieve_test.py

Step 7 — Run the API
uvicorn api:app --reload --port 8000

Step 8 — Test the API (PowerShell)
irm -Method Post "http://127.0.0.1:8000/recommend" `
-Headers @{ "Content-Type"="application/json" } `
-Body '{"query":"java developer with communication skills","top_k":5}'

Step 9 — Open the frontend
frontend/index.html(open live server)
4. Evaluation Result
Queries evaluated: 10
Mean Recall@10: 1.0
5. System Architecture
Excel Dataset
     ↓
Playwright / Data Fetch
     ↓
clean_catalog.py
     ↓
build_index.py  →  embeddings.npy + nn_model.joblib
     ↓
FastAPI backend (api.py)
     ↓
frontend/index.html (Browser UI)
7. Final Notes

The project avoids FAISS to eliminate Windows/NumPy binary issues.

MPNet embeddings provide strong semantic similarity.

FastAPI + frontend gives a complete working demo.

Evaluation confirms correctness and high quality.
