from openai import AzureOpenAI
from app.config import AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, LLM_MODEL

client = AzureOpenAI(
  api_key=AZURE_OPENAI_KEY,
  api_version="2024-02-15-preview",
  azure_endpoint=AZURE_OPENAI_ENDPOINT
)


def classify_intent(query):

  prompt = f"""
Classify the user query into one of the predefined intents.

Query: {query}

Only return the intent name.
"""

  try:
    res = client.chat.completions.create(
      model=LLM_MODEL,
      messages=[{"role": "user", "content": prompt}],
      temperature=0
    )

    return res.choices[0].message.content.strip()

  except Exception as e:
    print("LLM FAILED:", e)

    # 🔥 IMPORTANT: no crash
    return "fallback_intent"