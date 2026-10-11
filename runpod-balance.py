import json
import os
import sys
import urllib.request

api_key = os.environ.get("RUNPOD_API_KEY")
if not api_key:
    sys.exit("Error: RUNPOD_API_KEY environment variable is not set")

query = json.dumps({
    "query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"
})

url = f"https://api.runpod.io/graphql?api_key={api_key}"
req = urllib.request.Request(
    url,
    data=query.encode(),
    headers={
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP error {e.code}: {e.read().decode(errors='replace')}")

myself = data["data"]["myself"]
print(f"Balance:     ${myself['clientBalance']:.2f}")
print(f"Spend/hr:    ${myself['currentSpendPerHr']}")
print(f"Spend limit: ${myself['spendLimit']}")
