from openai import AzureOpenAI
from app.config import *
from intents.intent_phrases import INTENT_PHRASES

client = AzureOpenAI(
  api_key=AZURE_OPENAI_KEY,
  azure_endpoint=AZURE_OPENAI_ENDPOINT,
  api_version="2024-02-15-preview"
)

def classify_intent(query):
  prompt = f"""
  Classify the intent into one of these:
  {INTENT_PHRASES}

  Query: {query}

  Return only intent name.
  """

  res = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0
  )

  return res.choices[0].message.content.strip()