def split_intents(q):
  separators = [" and ", ",", " then ", " also "]

  for sep in separators:
    if sep in q:
      return [s.strip() for s in q.split(sep)]

  return [q]