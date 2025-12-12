MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.json"

# Options: flat | ivf | pq 
FAISS_MODE = "flat"

N_LIST = 5      # IVF cluster count
PQ_M = 16         # PQ subvector size
