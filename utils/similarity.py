from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(v1, v2):
  return cosine_similarity([v1], [v2])[0][0]