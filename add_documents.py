import json
import faiss
import numpy as np
from embedder import Embedder
from config import INDEX_PATH, METADATA_PATH

def add_documents(new_docs):
    # Load existing
    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Embed and add
    embedder = Embedder()
    new_emb = embedder.encode(new_docs)
    index.add(new_emb)

    # Update metadata
    metadata.extend(new_docs)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Documents added!")


if __name__ == "__main__":
    add_documents([
        "Neural networks learn complex patterns.",
        "Cosine similarity is angle-based matching."
    ])
