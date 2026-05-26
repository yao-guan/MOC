from sentence_transformers import SentenceTransformer
import os

model_path = "~/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_PATH = os.path.expanduser(model_path)

_model = None

def get_sentence_embedding(sentence):
    global _model
    if _model is None:
        _model = SentenceTransformer(LOCAL_MODEL_PATH)
    embeddings = _model.encode(sentence)
    return embeddings