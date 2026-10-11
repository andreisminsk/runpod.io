# RunPod Session Architecture

`runpod_session.py` — an ephemeral GPU pod lifecycle manager: rent a GPU under a
price cap, do work over SSH, tear it down, and report exactly what the session cost.

## Overview

One tool, two modes:

- **Granular** — individual subcommands (`balance`, `create`, `exec`, `send`,
  `recv`, `stop`, `terminate`, ...) for interactive use.
- **One-shot** — the `session` subcommand runs the entire lifecycle end-to-end
  and prints the session cost.

```
┌────────────────────┐   GraphQL (balance, gpuTypes)   ┌──────────────────┐
│  runpod_session.py │ ──────────────────────────────▶ │    RunPod API    │
│                    │   REST v2 (pod CRUD)            │  (Cloudflare-    │
│  balance           │ ──────────────────────────────▶ │   fronted)       │
│  gpu select ≤ cap  │                                 └──────────────────┘
│  create / wait     │
│  exec / send/recv  │   SSH + SFTP (paramiko)         ┌──────────────────┐
│  stop / terminate  │ ──────────────────────────────▶ │  GPU Pod         │
│  cost report       │                                 │  (your template) │
└─────────┬──────────┘                                 └──────────────────┘
          │
   .runpod_session.json  (last-pod state)
```

## Lifecycle (the `session` command)

| # | Step                | Function            | API                              |
|---|---------------------|---------------------|----------------------------------|
| 1 | Balance before      | `get_balance()`     | GraphQL `myself { clientBalance ... }` |
| 2 | GPU pick ≤ cap      | `pick_gpu()`        | GraphQL `gpuTypes { lowestPrice ... }` |
| 3 | Create pod          | `rest POST /pods`   | REST v2                          |
| 4 | Deploy wait         | `wait_running()`    | poll `GET /pods/{id}` until `RUNNING` + ports up |
| 5 | SSH work            | `ssh_exec()`, SFTP  | paramiko (uploads → script/commands → downloads) |
| 6 | Stop + terminate    | `rest` stop/DELETE  | REST v2                          |
| 7 | Cost report         | balance diff        | GraphQL `myself`                 |

## API surface

| Operation        | Endpoint / Query                                   | Auth               |
|------------------|----------------------------------------------------|--------------------|
| Balance          | `POST https://api.runpod.io/graphql` — `myself { clientBalance currentSpendPerHr spendLimit }` | `Authorization: Bearer` |
| GPU catalog      | same — `gpuTypes { id displayName memoryInGb lowestPrice { minimumBidPrice uninterruptablePrice } }` | same |
| Create pod       | `POST https://api.runpod.io/v2/pods`               | `?apikey=` param   |
| Pod status       | `GET  /v2/pods/{id}`                               | same               |
| Stop / start     | `POST /v2/pods/{id}/stop` · `/start`               | same               |
| Terminate        | `DELETE /v2/pods/{id}`                             | same               |

- **API key**: `RUNPOD_API_KEY` env var only — never hardcoded.
- **Cloudflare**: the default Python UA is blocked (error 1010); all requests
  send a browser-like `User-Agent`.

## GPU selection & price cap

0. If `--template` is given, fetch it (`podTemplate(id)`) and derive GPU
   compatibility from its docker image name (see next section).
1. Fetch all GPU types with their lowest prices.
2. Price depends on cloud type: `SECURE` → `uninterruptablePrice` (on-demand),
   `COMMUNITY` → `minimumBidPrice` (interruptible; `bidPerGpu` is set on create).
3. Filter to `price ≤ --max-price`.
4. If `--gpu <id|name>` is given, verify that specific type is within the cap
   (fail loudly if not). Otherwise pick the **cheapest**, ties broken by more VRAM.

### Template → GPU compatibility

Templates are resolved **by id or by name**: GraphQL `podTemplate(id:)` first;
if not found, your own templates via REST `GET /v2/templates` (Bearer auth),
matched on `name` — e.g. `--template image-gen1.0.0-cu128` resolves to id
`dfou0azyia`, image `ghcr.io/andreisminsk/image-gen:1.0.0-cu128`.

RunPod templates don't expose a supported-GPU list via the API, so compatibility
is derived from the image name tag (e.g. `image-gen1.0.0-cu128`):

| Image tag             | Meaning         | Compatible GPUs                                  |
|-----------------------|-----------------|--------------------------------------------------|
| `cu128`+              | CUDA 12.8+      | all NVIDIA incl. Blackwell (cc ≤ 12.0)           |
| `cu118`–`cu127`       | CUDA 11.8–12.7  | NVIDIA cc ≤ 9.0 (Hopper; **no** Blackwell)       |
| `cu111`–`cu117`       | CUDA 11.1–11.7  | NVIDIA cc ≤ 8.6 (Ampere)                         |
| `cu110`               | CUDA 11.0       | NVIDIA cc ≤ 8.0                                  |
| `rocm`                | ROCm            | AMD (MI300X, MI350X) only                        |
| no tag                | unknown         | no restriction (all GPUs)                        |

Additionally, template **compatibility settings** from REST `/v2/templates` are
enforced when present:

- `category` (`NVIDIA`/`AMD`) — restricts manufacturer.
- `allowedCudaVersions` (e.g. `["12.8"]`) — GPU must support at least the max
  listed CUDA version. Each GPU family has a minimum CUDA it can run
  (Blackwell→12.8, Hopper/Ada→11.8, Ampere GA102→11.1, GA100→11.0,
  Turing→10.0, Volta→9.0, Pascal→8.0).
- `disk` — minimum container disk; `create`/`session` auto-bump
  `--container-disk` if below (e.g. 100 GB for `image-gen1.0.0-cu128`).
- Template `ports`/`env` are used automatically when creating from a template
  (explicit `--ports`/`--env` override).

The API exposes **no min-VRAM field** on templates, so use `--min-vram <GB>`
to enforce it manually (applies to `gpus`, `create`, and `session`).

Compute capability per family (well-known constants, matched on GPU id/name):
Blackwell 12.0 (RTX 50xx, RTX PRO), 10.0 (B200/B300), Hopper 9.0 (H100/H200),
Ada 8.9 (RTX 40xx, L4, L40S, *Ada), Ampere 8.6 (RTX 30xx, A40, RTX Ax000),
8.0 (A100/A30), Turing 7.5 (T4), Volta 7.0 (V100), Pascal 6.0.

Example: `image-gen1.0.0-cu128` → CUDA 12.8 → any NVIDIA GPU within the price
cap (including Blackwell); AMD cards excluded. `gpus --template <id>` lists
only compatible GPUs.

## SSH layer

- Pod must expose `22/tcp` (default `--ports 22/tcp,8888/http`).
- Auth: template must receive your public key, typically via
  `--env PUBLIC_KEY='ssh-ed25519 ...'` (or a template that already has it).
  Password auth is supported as a fallback (`--password`).
- Endpoint resolution: after `RUNNING`, `GET /pods/{id}` → `runtime.ports`;
  find `privatePort == 22`, map to `publicPort`; host = `pod.hostname`
  (proxy) or port `ip` (direct-IP pods). User is `root`.
- Transport: `paramiko` — `exec_command()` for commands (exit code propagated),
  SFTP `put`/`get` for file transfer. Key default: `~/.ssh/id_rsa`.

## State

`.runpod_session.json` (working directory) stores the last created pod so
`exec`/`send`/`recv`/`stop`/`terminate` accept `last` (or no argument):

```json
{
  "pod_id": "abc123",
  "name": "ephemeral-20261012-101500",
  "gpu": "NVIDIA GeForce RTX 4090",
  "cost_per_hr": 0.34,
  "template": "tpl-xyz",
  "created_at": "2026-10-12T10:15:00+00:00",
  "ssh_host": "proxy.runpod.net",
  "ssh_port": 12345
}
```

Deleted on `terminate` (unless `session --keep`).

## Cost accounting

- **Session cost = balance_before − balance_after** (authoritative, includes
  all billing: GPU, disk, network).
- Sanity check: `costPerHour × uptime` should be close to the diff.
- RunPod balance updates can lag a few seconds after termination — the script
  waits 5s before the final read.

## Error handling

- All API failures → `die()` with HTTP status + response body excerpt.
- Deploy timeout (`--wait-timeout`, default 600s) → error reminding you the pod
  may still be billing; state file is already saved so `terminate last` works.
- `exec` propagates the remote exit code as the script's exit code.
- Script failure inside `session` prints a warning but **still tears the pod
  down** — you never pay for a forgotten pod.

## Security notes

- API key via environment only; never logged or written to state.
- `AutoAddPolicy` accepts unknown host keys (TOFU) — acceptable for short-lived
  ephemeral pods; tighten if you pin hosts.
- Community-cloud pods are interruptible; use `SECURE` for reliability.

## Usage

```bash
export RUNPOD_API_KEY=...

python runpod_session.py balance
python runpod_session.py gpus --max-price 2.0
python runpod_session.py gpus --max-price 2.0 --template tpl-xxxxxxxx  # compatible only

# One-shot session: cheapest GPU ≤ $2/hr, run a script, fetch results
python runpod_session.py session \
  --template tpl-xxxxxxxx \
  --max-price 2.0 \
  --env PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  --send ./input.bin:/workspace/input.bin \
  --script ./job.sh \
  --recv /workspace/output.bin:./output.bin

# Granular
python runpod_session.py create --template tpl-xxx --max-price 2.0
python runpod_session.py exec last "nvidia-smi"
python runpod_session.py send last model.pt /workspace/model.pt
python runpod_session.py stop last
python runpod_session.py terminate last
```

## Dependencies & limitations

- `pip install requests paramiko`; Python 3.8+.
- Single-pod state file (one concurrent session per working directory).
- `exec` has no hard wall-clock kill for commands that ignore timeouts.
- Balance diff may include residual billing from other running pods — check
  `Spend/hr` is $0 before starting a session for a clean measurement.
