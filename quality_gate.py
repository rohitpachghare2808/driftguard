import json
import sys

f1 = json.load(open("metrics.json"))["f1"]
THRESHOLD = 1.01
print(f"F1={f1:.3f}, threshold={THRESHOLD}")
sys.exit(0 if f1 >= THRESHOLD else 1)
