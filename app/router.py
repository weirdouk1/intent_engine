import time
from intents.corpus import CORPUS
from utils.preprocessing import clean_text
from models.embedding_model import get_embedding
from models.vector_store import search
from models.llm_classifier import classify_intent
from app.config import SIMILARITY_THRESHOLD


# 🔹 INTENT DISAMBIGUATION GROUPS
FLIGHT_STATUS_WORDS = ["when", "time", "schedule", "departure", "arrival", "next"]
FLIGHT_DELAY_WORDS = ["delay", "delayed", "late"]
BOARDING_WORDS = ["boarding", "gate", "checkin"]


def detect_intent(query):
  start_time = time.time()

  q = clean_text(query)
  words = q.split()

  # 🔹 RULE-BASED (FASTEST)
  for k in CORPUS:
    key_words = k.split()

    if all(w in words for w in key_words):
      return {
        "intent": CORPUS[k],
        "method": "rule",
        "score": None,
        "latency": time.time() - start_time
      }

  # 🔹 EMBEDDING (MAIN ENGINE)
  emb_start = time.time()

  vec = get_embedding(q)
  intent, score = search(vec)

  emb_time = time.time() - emb_start

  print("Similarity score:", score)

  # 🔹 INTENT DISAMBIGUATION (SMART LAYER)
  if intent in ["flight_delay", "flight_status", "boarding_info"]:

    if any(w in q for w in FLIGHT_DELAY_WORDS):
      intent = "flight_delay"

    elif any(w in q for w in BOARDING_WORDS):
      intent = "boarding_info"

    elif any(w in q for w in FLIGHT_STATUS_WORDS):
      intent = "flight_status"

  # 🔹 EMBEDDING DECISION
  if score < SIMILARITY_THRESHOLD:
    return {
      "intent": intent,
      "method": "embedding",
      "score": score,
      "latency": time.time() - start_time,
      "embedding_time": emb_time,
      "llm_time": 0
    }

  # 🔹 LLM FALLBACK (RARE)
  llm_start = time.time()

  intent = classify_intent(q)

  llm_time = time.time() - llm_start

  return {
    "intent": intent,
    "method": "llm",
    "score": score,
    "latency": time.time() - start_time,
    "embedding_time": emb_time,
    "llm_time": llm_time
  }