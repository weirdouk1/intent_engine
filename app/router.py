import time
from intents.corpus import CORPUS
from utils.preprocessing import clean_text
from models.embedding_model import get_embedding
from models.vector_store import search
from models.llm_classifier import classify_intent

SIMILARITY_THRESHOLD = 0.7   # relaxed
LLM_TRIGGER_THRESHOLD = 1.2  # only very bad matches


FLIGHT_STATUS_WORDS = ["when", "time", "schedule", "departure", "arrival", "next"]
FLIGHT_DELAY_WORDS = ["delay", "delayed", "late"]
BOARDING_WORDS = ["boarding", "gate", "checkin"]


def detect_intent(query):
  start_time = time.time()

  q = clean_text(query)
  words = q.split()

  # 🔹 RULE LAYER
  for k in CORPUS:
    key_words = k.split()
    if all(w in words for w in key_words):
      return {
        "intent": CORPUS[k],
        "method": "rule",
        "score": None,
        "latency": time.time() - start_time
      }

  # 🔹 EMBEDDING
  emb_start = time.time()

  vec = get_embedding(q)
  results = search(vec, k=3)

  emb_time = time.time() - emb_start

  best_intent, best_score = results[0]
  second_intent, second_score = results[1]

  print("Top results:", results)

  # 🔹 DOMAIN FIX (VERY IMPORTANT)
  if best_intent in ["flight_delay", "flight_status", "boarding_info"]:

    if any(w in q for w in FLIGHT_DELAY_WORDS):
      best_intent = "flight_delay"

    elif any(w in q for w in BOARDING_WORDS):
      best_intent = "boarding_info"

    elif any(w in q for w in FLIGHT_STATUS_WORDS):
      best_intent = "flight_status"

  # 🔥 NEW DECISION LOGIC

  # ✅ GOOD MATCH → USE EMBEDDING
  if best_score < SIMILARITY_THRESHOLD:
    return {
      "intent": best_intent,
      "method": "embedding",
      "score": best_score,
      "latency": time.time() - start_time,
      "embedding_time": emb_time,
      "llm_time": 0
    }

  # ⚠️ MEDIUM MATCH → STILL USE EMBEDDING (NO LLM!)
  if best_score < LLM_TRIGGER_THRESHOLD:
    return {
      "intent": best_intent,
      "method": "embedding_soft",
      "score": best_score,
      "latency": time.time() - start_time,
      "embedding_time": emb_time,
      "llm_time": 0
    }

  # 🚨 ONLY VERY BAD → LLM
  intent = classify_intent(q)

  return {
    "intent": intent,
    "method": "llm",
    "score": best_score,
    "latency": time.time() - start_time,
    "embedding_time": emb_time
  }