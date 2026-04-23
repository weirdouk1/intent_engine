LLM_CACHE = {}

def get_cache(q):
  return LLM_CACHE.get(q)

def set_cache(q, intent):
  LLM_CACHE[q] = intent