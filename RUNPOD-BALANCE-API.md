# Runpod: Get Current Balance

## 1. GraphQL API (programmatic)

```bash
curl --request POST \
  --header 'content-type: application/json' \
  --url 'https://api.runpod.io/graphql?api_key=${YOUR_API_KEY}' \
  --data '{"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"}'
```

The `myself` query returns:
- `clientBalance` — prepaid credit balance
- `currentSpendPerHr` — current hourly spend
- `spendLimit` — account spend limit

## Windows

### cmd.exe

```cmd
curl --request POST ^
  --header "content-type: application/json" ^
  --url "https://api.runpod.io/graphql?api_key=YOUR_API_KEY" ^
  --data "{\"query\": \"query { myself { clientBalance currentSpendPerHr spendLimit } }\"}"
```

### PowerShell (curl.exe)

Use `curl.exe` — plain `curl` is an alias for `Invoke-WebRequest` in Windows PowerShell 5.1.

```powershell
curl.exe --request POST `
  --header 'content-type: application/json' `
  --url 'https://api.runpod.io/graphql?api_key=YOUR_API_KEY' `
  --data '{"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"}'
```

### PowerShell native (Invoke-RestMethod)

```powershell
Invoke-RestMethod -Method Post `
  -Uri 'https://api.runpod.io/graphql?api_key=YOUR_API_KEY' `
  -ContentType 'application/json' `
  -Body '{"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"}'
```

## Python (API key from environment)

```python
import json
import os
import urllib.request

api_key = os.environ["RUNPOD_API_KEY"]

query = json.dumps({
    "query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"
})

url = f"https://api.runpod.io/graphql?api_key={api_key}"
req = urllib.request.Request(
    url,
    data=query.encode(),
    headers={
        "Content-Type": "application/json",
        # Cloudflare blocks the default urllib User-Agent (error 1010)
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    },
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    data = json.load(resp)

myself = data["data"]["myself"]
print(f"Balance: ${myself['clientBalance']:.2f}")
print(f"Spend/hr: ${myself['currentSpendPerHr']}")
print(f"Spend limit: ${myself['spendLimit']}")
```

## 2. runpodctl CLI

```bash
runpodctl user    # or: runpodctl me
```

Returns JSON including `clientBalance`, `currentSpendPerHr`, `spendLimit`, email, and notification settings.

## 3. Console (manual)

Billing page: https://www.console.runpod.io/user/billing — shows balance and recent transactions.

## Notes

- Post-paid accounts don't display a balance (billing is via cost centers instead).
- Pods are automatically stopped when balance reaches $0.
