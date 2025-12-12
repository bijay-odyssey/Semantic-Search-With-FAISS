import json
import faiss
import numpy as np
from embedder import Embedder
from config import INDEX_PATH, METADATA_PATH

# Global variables
index = None
metadata = None
embedder = Embedder()


# Load / Reload FAISS Index
def load_index():
    global index, metadata

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"[INFO] Index loaded. Vectors: {index.ntotal}, Metadata: {len(metadata)}")
    except Exception as e:
        print("[WARNING] Could not load index:", e)
        index = None
        metadata = []


# Load index initially
load_index()


# Search Function
def search(query, k=5):
    if index is None or metadata is None:
        return [{"error": "Index not built yet."}]

    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            results.append({"text": metadata[idx], "score": float(score)})

    return results
