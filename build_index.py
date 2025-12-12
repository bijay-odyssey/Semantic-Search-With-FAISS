import json
import numpy as np
import faiss
from embedder import Embedder
from config import INDEX_PATH, METADATA_PATH, FAISS_MODE, N_LIST, PQ_M

def build_index(texts):
    embedder = Embedder()
    embeddings = embedder.encode(texts)
    d = embeddings.shape[1]

    # Choose index type
    if FAISS_MODE == "flat":
        index = faiss.IndexFlatIP(d)

    elif FAISS_MODE == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, N_LIST)
        print("Training IVF...")
        index.train(embeddings)

    elif FAISS_MODE == "pq":
        index = faiss.IndexPQ(d, PQ_M, 8)
        print("Training PQ...")
        index.train(embeddings)

    else:
        raise ValueError("Invalid FAISS_MODE")

    # Add vectors
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_PATH)

    # Save metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(texts, f, indent=2)

    print(f"Index saved: {INDEX_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    documents = [
        "FAISS supports large-scale vector search.",
        "Semantic search finds similar meaning.",
        "Transformers create embeddings for text.",
        "Pizza tastes great on weekends.",
        "Vector databases store high-dimensional data.",
    ]

    build_index(documents)
