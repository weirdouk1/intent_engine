from app.router import detect_intent
from app.logger import get_logger
from intents.handlers.booking_handler import handle_booking
from intents.handlers.complaint_handler import handle_complaint
from intents.handlers.info_handler import handle_info

logger = get_logger()

from dotenv import load_dotenv
load_dotenv()

def route_to_handler(intent):
  if intent in ["flight_booking", "hotel_booking"]:
    return handle_booking(intent)
  elif intent in ["complaint", "feedback"]:
    return handle_complaint(intent)
  else:
    return handle_info(intent)

if __name__ == "__main__":
  while True:
    q = input("Query: ")
    if not q.strip():
      print("Empty query, please enter something.")
      continue

    result = detect_intent(q)
    intent = result["intent"]
    method = result["method"]
    score = result["score"]
    latency = result["latency"]

    response = route_to_handler(intent)

    print(f"Intent: {intent}")
    print(f"Method: {method}")
    print(f"Similarity Score: {score}")
    print(f"Latency: {latency:.3f} sec")
    print(f"Response: {response}")