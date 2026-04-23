import sys
import os

# 🔥 FORCE ROOT PATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
from app.router import detect_intent

st.set_page_config(page_title="Intent Recognition Engine", layout="centered")

st.title("🧠 Intent Recognition Engine")
st.markdown("Enterprise-level Intent Mapping Demo")


# 🔹 function to process query
def process_query():
  query = st.session_state.query

  if query.strip() == "":
    st.warning("Please enter a query")
    return

  result = detect_intent(query)

  st.session_state.result = result


# 🔹 INPUT (ENTER triggers this)
st.text_input(
  "Enter your query:",
  key="query",
  on_change=process_query  # 🔥 ENTER triggers this
)

# 🔹 BUTTON (optional backup)
if st.button("Detect Intent"):
  process_query()


# 🔹 SHOW RESULT
if "result" in st.session_state:
  result = st.session_state.result

  st.divider()

  col1, col2 = st.columns(2)

  with col1:
    st.metric("Intent", result["intent"])
    st.metric("Method", result["method"])

  with col2:
    st.metric("Latency (sec)", f"{result['latency']:.3f}")
    st.metric(
      "Similarity",
      "N/A" if result["score"] is None else f"{result['score']:.3f}"
    )

  st.divider()

  st.subheader("Response")
  st.success(result.get("response", f"Handled intent: {result['intent']}"))


st.markdown("---")
st.caption("⚡ Built using Hybrid AI (Rule + Embedding + LLM)")