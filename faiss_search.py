import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Texts to be indexed
texts = [
    "FAISS is a library for efficient similarity search.",
    "The sky is bright and blue.",
    "Machine learning helps computers learn patterns.",
    "Neural networks are used for deep learning.",
    "Guitar is a popular musical instrument.",
    "Vector databases store high-dimensional vectors.",
    "Cosine similarity measures angle between vectors.",
    "I love eating pizza on weekends.",
    "Python is great for prototyping AI systems.",
    "Semantic search finds meaning, not keywords."
]

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Normalize embeddings for cosine similarity
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

d = emb_norm.shape[1]   # dimension

# Build FAISS index (Cosine via Inner Product)
index = faiss.IndexFlatIP(d)  

# Add vectors to FAISS
index.add(emb_norm)

print(f"Number of vectors indexed: {index.ntotal}")

# Search function to find top-k similar texts
def search(query, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    scores, indices = index.search(q_emb, k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append((texts[idx], float(score)))

    return results

# CLI for querying
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or 'exit'): ")

        if query == "exit":
            break

        print("\nTop results:")
        for chunk, score in search(query, k=3):
            print(f"{score:.4f} | {chunk}")
