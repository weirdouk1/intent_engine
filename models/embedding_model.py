from sentence_transformers import SentenceTransformer
from models.cache import get_cache, set_cache

model = SentenceTransformer('all-mpnet-base-v2')

def get_embedding(text):
  key = text.lower().strip()

  cached = get_cache(key)
  if cached:
    return cached

  vec = model.encode(key).tolist()
  set_cache(key, vec)

  return vec