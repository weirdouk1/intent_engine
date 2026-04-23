from rapidfuzz import process

def fuzzy_match(query, corpus_keys, threshold=85):
  result = process.extractOne(query, corpus_keys)

  if result:
    match, score, _ = result

    if score >= threshold:
      return match

  return None