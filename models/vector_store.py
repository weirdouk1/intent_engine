import faiss
import numpy as np
import os
from intents.intent_phrases import INTENT_PHRASES
from models.embedding_model import get_embedding

DIMENSION = 768

index = faiss.IndexFlatL2(DIMENSION)
intent_labels = []


def init_index():
  global index, intent_labels

  if os.path.exists("faiss.index"):
    print("Loading FAISS index...")
    index = faiss.read_index("faiss.index")

    # 🔥 rebuild labels
    intent_labels.clear()

    for intent, phrases in INTENT_PHRASES.items():
      for _ in phrases:
        intent_labels.append(intent)

    return

  print("Creating FAISS index...")

  for intent, phrases in INTENT_PHRASES.items():
    for phrase in phrases:
      vec = get_embedding(phrase)
      vec = np.array(vec).astype("float32")

      # 🔥 NORMALIZATION (IMPORTANT)
      faiss.normalize_L2(vec.reshape(1, -1))

      index.add(vec.reshape(1, -1))
      intent_labels.append(intent)

  faiss.write_index(index, "faiss.index")


def search(vec):
  vec = np.array(vec).astype("float32")

  # 🔥 NORMALIZATION (IMPORTANT)
  faiss.normalize_L2(vec.reshape(1, -1))

  D, I = index.search(vec.reshape(1, -1), 1)

  return intent_labels[I[0][0]], D[0][0]


init_index()