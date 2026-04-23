import time
from intents.corpus import CORPUS
from utils.preprocessing import clean_text
from models.embedding_model import get_embedding
from models.vector_store import search
from models.llm_classifier import classify_intent

from utils.multi_intent import split_intents
from utils.fuzzy import fuzzy_match
from models.cache import get_cache, set_cache
from models.auto_memory import get_memory, set_memory


# 🔥 UPDATED THRESHOLDS (LESS LLM)
SIMILARITY_THRESHOLD = 0.9
LLM_TRIGGER_THRESHOLD = 1.6


FLIGHT_STATUS_WORDS = ["when", "time", "schedule", "departure", "arrival", "next"]
FLIGHT_DELAY_WORDS = ["delay", "delayed", "late"]
BOARDING_WORDS = ["boarding", "gate", "checkin"]


# 🔹 SINGLE INTENT
def detect_single_intent(q):

  words = q.split()

  # 🔥 1. AUTO MEMORY
  mem = get_memory(q)
  if mem:
    return mem, "memory", None

  # 🔹 2. RULE
  for k in CORPUS:
    key_words = k.split()
    if all(w in words for w in key_words):
      return CORPUS[k], "rule", None

  # 🔹 3. EMOTION RULE
  if any(w in q for w in ["bad", "worst", "hate", "confusing"]):
    return "complaint", "rule_emotion", None

  # 🔹 4. DOMAIN RULE (NEW 🔥)
  if any(w in q for w in ["water", "bottle", "liquid"]):
    return "baggage_info", "rule_liquid", None

  # 🔹 5. FUZZY (SHORT ONLY)
  if len(words) <= 3:
    match = fuzzy_match(q, list(CORPUS.keys()))
    if match:
      return CORPUS[match], "fuzzy", None

  # 🔹 6. EMBEDDING
  vec = get_embedding(q)
  results = search(vec, k=3)

  best_intent, best_score = results[0]

  print("Top results:", results)

  # 🔥 7. SEMANTIC BOOST
  if any(w in q for w in ["luggage", "baggage", "carry", "weight"]):
    best_intent = "baggage_info"

  # 🔹 8. DOMAIN FIX
  if any(w in q for w in FLIGHT_DELAY_WORDS):
    best_intent = "flight_delay"

  elif any(w in q for w in BOARDING_WORDS):
    best_intent = "boarding_info"

  elif any(w in q for w in FLIGHT_STATUS_WORDS):
    best_intent = "flight_status"

  # 🔹 9. DECISION

  if best_score < SIMILARITY_THRESHOLD:
    return best_intent, "embedding", best_score

  if best_score < LLM_TRIGGER_THRESHOLD:
    return best_intent, "embedding_soft", best_score

  # 🔹 10. CACHE
  cached = get_cache(q)
  if cached:
    return cached, "cache", best_score

  # 🔹 11. LLM
  intent = classify_intent(q)

  # 🔥 FAIL SAFE
  if intent == "fallback_intent":
    return best_intent, "embedding_fallback", best_score

  set_cache(q, intent)

  return intent, "llm", best_score


# 🔹 MAIN
def detect_intent(query):

  start_time = time.time()

  q = clean_text(query)
  parts = split_intents(q)

  results = []
  need_llm = False

  for part in parts:

    intent, method, score = detect_single_intent(part)

    if method == "llm":
      need_llm = True

    # 🔥 AUTO LEARNING
    if method not in ["llm"]:
      set_memory(part, intent)

    results.append({
      "intent": intent,
      "method": method,
      "score": score,
      "query": part
    })

  # 🔥 GLOBAL LLM (ONLY ONCE)
  if need_llm:
    intent = classify_intent(q)

    if intent == "fallback_intent":
      intent = results[0]["intent"]  # safe fallback

    set_memory(q, intent)

    return {
      "results": [{
        "intent": intent,
        "method": "llm_global",
        "score": None,
        "query": q
      }],
      "latency": time.time() - start_time
    }

  return {
    "results": results,
    "latency": time.time() - start_time
  }