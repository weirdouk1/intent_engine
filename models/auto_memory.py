import json
import os

MEMORY_FILE = "data/auto_memory.json"

# ensure folder exists
os.makedirs("data", exist_ok=True)

if os.path.exists(MEMORY_FILE):
  try:
    with open(MEMORY_FILE, "r") as f:
      MEMORY = json.load(f)
  except:
    MEMORY = {}
else:
  MEMORY = {}


def get_memory(q):
  return MEMORY.get(q)


def set_memory(q, intent):
  MEMORY[q] = intent

  with open(MEMORY_FILE, "w") as f:
    json.dump(MEMORY, f, indent=2)