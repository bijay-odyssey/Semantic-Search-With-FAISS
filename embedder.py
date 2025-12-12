import numpy as np
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts):
        emb = self.model.encode(texts, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype("float32")
