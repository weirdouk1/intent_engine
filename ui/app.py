import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.router import detect_intent

st.set_page_config(page_title="Intent Engine", layout="centered")

st.title("🧠 Intent Recognition Engine")
st.caption("Hybrid AI • Fast • Smart Routing")


# 🔹 PROCESS FUNCTION
def process_query():

  query = st.session_state.get("query", "")

  if query.strip() == "":
    st.warning("Please enter a query")
    return

  result = detect_intent(query)
  st.session_state.result = result


# 🔹 INPUT (NO on_change → avoids double execution ❌)
st.text_input(
  "Type your query:",
  key="query"
)


# 🔹 BUTTON (ONLY trigger)
if st.button("Detect Intent"):
  process_query()


# 🔹 DISPLAY RESULTS
if "result" in st.session_state:

  result = st.session_state.result

  st.divider()

  st.subheader("🧾 Results")

  for i, r in enumerate(result["results"], 1):

    with st.container():
      st.markdown(f"### 🔹 Intent {i}")

      col1, col2 = st.columns(2)

      with col1:
        st.metric("Intent", r["intent"])
        st.metric("Method", r["method"])

      with col2:
        st.metric(
          "Score",
          "N/A" if r["score"] is None else f"{r['score']:.3f}"
        )

      st.markdown(f"**Query part:** `{r['query']}`")

      st.markdown("---")

  # 🔹 LATENCY
  st.success(f"⚡ Total Latency: {result['latency']:.3f} sec")


st.markdown("---")
st.caption("🚀 Hybrid: Rule + Fuzzy + Embedding + LLM")