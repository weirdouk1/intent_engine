local_cache = {}

def get_cache(key):
  return local_cache.get(key)

def set_cache(key, value):
  local_cache[key] = value