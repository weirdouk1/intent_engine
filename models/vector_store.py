import faiss
import numpy as np
import os
from intents.intent_phrases import INTENT_PHRASES
from models.embedding_model import get_embedding

DIMENSION = 384

index = faiss.IndexFlatL2(DIMENSION)
intent_labels = []


def init_index():
  global index, intent_labels

  if os.path.exists("faiss.index"):
    print("Loading FAISS index...")
    index = faiss.read_index("faiss.index")

    intent_labels.clear()
    for intent, phrases in INTENT_PHRASES.items():
      for _ in phrases:
        intent_labels.append(intent)

    return

  print("Creating FAISS index...")

  for intent, phrases in INTENT_PHRASES.items():
    for phrase in phrases:
      vec = np.array(get_embedding(phrase)).astype("float32")

      faiss.normalize_L2(vec.reshape(1, -1))

      index.add(vec.reshape(1, -1))
      intent_labels.append(intent)

  faiss.write_index(index, "faiss.index")


def search(vec, k=3):
  vec = np.array(vec).astype("float32")

  D, I = index.search(np.array([vec]), k)

  results = []
  for i in range(k):
    idx = I[0][i]
    if idx < len(intent_labels):
      results.append((intent_labels[idx], float(D[0][i])))

  return results

init_index()