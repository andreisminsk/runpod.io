#!/usr/bin/env python3
"""
runpod_session.py — ephemeral RunPod GPU pod lifecycle manager.

Subcommands:
  balance     Show account balance / spend rate / limit
  gpus        List GPU types with prices (optionally filtered by --max-price)
  create      Create + deploy a pod from a template or docker image
  status      Show pod status
  exec        Run a command on the pod over SSH
  send        Upload a file to the pod (SFTP)
  recv        Download a file from the pod (SFTP)
  stop        Stop the pod
  start       Start a stopped pod
  terminate   Terminate (delete) the pod
  session     Full lifecycle: balance -> create -> deploy -> SSH work ->
              stop -> terminate -> balance + session cost

Requires:
  RUNPOD_API_KEY env var (https://www.runpod.io/console/user/settings)
  pip install requests paramiko
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is required: pip install requests")

API_BASE = "https://api.runpod.io"
GRAPHQL_URL = f"{API_BASE}/graphql"
REST_URL = f"{API_BASE}/v2"
STATE_FILE = Path(".runpod_session.json")
SSH_USER = "root"
DEFAULT_PORTS = "22/tcp,8888/http"
CLOUD_CHOICES = ["SECURE", "COMMUNITY", "ALL"]

# Cloudflare blocks the default python-requests UA (error 1010) — use a browser-like one.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def get_api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        die("RUNPOD_API_KEY environment variable is not set "
            "(get one at https://www.runpod.io/console/user/settings)")
    return key


def money(v):
    return f"${v:.2f}" if isinstance(v, (int, float)) else "n/a"


# ----------------------------------------------------------------- HTTP layer

def graphql(query, variables=None):
    r = requests.post(
        GRAPHQL_URL,
        headers={**HEADERS, "Authorization": f"Bearer {get_api_key()}"},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    if r.status_code != 200:
        die(f"GraphQL HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if data.get("errors"):
        die(f"GraphQL error: {json.dumps(data['errors'])[:300]}")
    return data["data"]


def rest(method, path, payload=None):
    r = requests.request(
        method,
        f"{REST_URL}{path}",
        headers={**HEADERS, "Authorization": f"Bearer {get_api_key()}"},
        params={"apikey": get_api_key()},
        json=payload,
        timeout=60,
    )
    if r.status_code not in (200, 201):
        die(f"REST {method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError:
        return r.text


# ------------------------------------------------------------------- balance

BALANCE_QUERY = "query { myself { clientBalance currentSpendPerHr spendLimit } }"


def get_balance():
    return graphql(BALANCE_QUERY)["myself"]


# ------------------------------------------------------------- GPU selection

GPU_TYPES_QUERY = """
query {
  gpuTypes {
    id
    displayName
    manufacturer
    memoryInGb
    lowestPrice { minimumBidPrice uninterruptablePrice }
  }
}
"""


def gpu_price(gpu, cloud_type):
    lp = gpu.get("lowestPrice") or {}
    if cloud_type == "COMMUNITY":
        return lp.get("minimumBidPrice")
    return lp.get("uninterruptablePrice")


# --- template -> GPU compatibility ------------------------------------------
# Compute capability by GPU family (well-known constants, matched on id/name).
GPU_CC_RULES = [
    (r"RTX 50\d\d|RTX PRO|PRO 6000|PRO 5000", 12.0),   # Blackwell (sm_120)
    (r"\bB300\b|\bB200\b|\bGB200", 10.0),              # Blackwell datacenter
    (r"H200|H100|GH200|\bH20\b", 9.0),                 # Hopper
    (r"RTX 40\d\d|L40|L4\b|Ada", 8.9),                 # Ada Lovelace
    (r"A40|RTX A\d", 8.6),                             # Ampere (GA102)
    (r"RTX 30\d\d", 8.6),                              # Ampere consumer
    (r"A100|A30", 8.0),                                # Ampere (GA100)
    (r"\bT4\b", 7.5),                                  # Turing
    (r"V100", 7.0),                                    # Volta
    (r"P40|P100", 6.0),                                # Pascal
    (r"MI\d", None),                                   # AMD Instinct -> ROCm
]

TEMPLATE_QUERY = """
query ($id: String!) {
  podTemplate(id: $id) { id name imageName }
}
"""


def get_template(template_ref):
    """Resolve a template by id or name. Own templates via REST /v2/templates
    (rich fields: category, allowedCudaVersions, disk, ports, env) first;
    GraphQL podTemplate (public templates) as fallback."""
    r = requests.get(f"{REST_URL}/templates",
                     headers={**HEADERS, "Authorization": f"Bearer {get_api_key()}"},
                     timeout=30)
    if r.status_code == 200:
        for t in r.json().get("templates") or []:
            if template_ref in (t.get("id"), t.get("name")):
                return {"id": t["id"], "name": t.get("name"),
                        "imageName": t.get("image"),
                        "category": t.get("category") or "",
                        "allowedCudaVersions": t.get("allowedCudaVersions") or [],
                        "disk": t.get("disk") or 0,
                        "ports": t.get("ports") or [],
                        "env": t.get("env") or {}}
    d = graphql(TEMPLATE_QUERY, {"id": template_ref})
    tpl = d.get("podTemplate")
    if tpl:
        return {"id": tpl["id"], "name": tpl.get("name"),
                "imageName": tpl.get("imageName"), "category": "",
                "allowedCudaVersions": [], "disk": 0, "ports": [], "env": {}}
    die(f"Template '{template_ref}' not found or not accessible")


def gpu_cc(gpu):
    text = f"{gpu.get('id', '')} {gpu.get('displayName', '')}"
    for pat, cc in GPU_CC_RULES:
        if re.search(pat, text, re.IGNORECASE):
            return cc
    return None


def parse_image_compat(image_name):
    """Derive GPU compatibility from a template's docker image name.
    e.g. 'image-gen1.0.0-cu128' -> CUDA 12.8 -> Blackwell-capable (max cc 12.0)."""
    name = (image_name or "").lower()
    if "rocm" in name:
        return {"kind": "rocm", "image": image_name}
    m = re.search(r"cu(\d{3})", name)  # first 3 digits: cu1281 -> cu128
    if m:
        cuda = int(m.group(1))
        if cuda >= 128:
            max_cc = 12.0
        elif cuda >= 118:
            max_cc = 9.0
        elif cuda >= 111:
            max_cc = 8.6
        elif cuda >= 110:
            max_cc = 8.0
        elif cuda >= 100:
            max_cc = 7.5
        else:
            max_cc = 7.0
        return {"kind": "cuda", "cuda": f"{cuda // 10}.{cuda % 10}",
                "cuda_float": cuda / 10.0, "max_cc": max_cc, "image": image_name}
    return {"kind": "unknown", "image": image_name}


# Minimum CUDA version each GPU family can run (GPU supports CUDA V iff V >= min).
MIN_CUDA_FOR_CC = [
    (12.0, 12.8),  # Blackwell
    (10.0, 12.8),  # Blackwell datacenter
    (9.0, 11.8),   # Hopper
    (8.9, 11.8),   # Ada
    (8.6, 11.1),   # Ampere GA102
    (8.0, 11.0),   # Ampere GA100
    (7.5, 10.0),   # Turing
    (7.0, 9.0),    # Volta
    (6.0, 8.0),    # Pascal
]


def gpu_min_cuda(gpu):
    cc = gpu_cc(gpu)
    if cc is None:
        return None
    for floor, min_cuda in MIN_CUDA_FOR_CC:
        if cc >= floor:
            return min_cuda
    return 12.9  # unknown newer family: assume latest CUDA


def describe_compat(compat):
    cat = (compat.get("category") or "").upper()
    if compat["kind"] == "rocm" or cat == "AMD":
        return "ROCm/AMD GPUs only (template category: AMD)"
    if compat["kind"] == "cuda":
        return (f"CUDA {compat['cuda']} image -> NVIDIA GPUs with compute "
                f"capability <= {compat['max_cc']}")
    if cat == "NVIDIA":
        return "NVIDIA GPUs only (template category)"
    return "no GPU restriction from image tag"


def gpu_compatible(gpu, compat):
    if not compat:
        return True
    mfr = (gpu.get("manufacturer") or "").lower()
    cat = (compat.get("category") or "").upper()
    if compat["kind"] == "rocm" or cat == "AMD":
        return mfr == "amd"
    if (compat["kind"] == "cuda" or cat == "NVIDIA") and mfr != "nvidia":
        return False
    allowed = compat.get("allowed_cuda")
    if allowed:
        mc = gpu_min_cuda(gpu)
        return mc is not None and max(allowed) >= mc
    if compat["kind"] == "cuda":
        mc = gpu_min_cuda(gpu)
        return mc is not None and compat["cuda_float"] >= mc
    return True


def template_compat(template_ref):
    """Fetch the template and derive its GPU compatibility (or None).
    Merges template settings (category, allowedCudaVersions, disk) with
    image-tag-derived rules."""
    if not template_ref:
        return None
    tpl = get_template(template_ref)
    compat = parse_image_compat(tpl.get("imageName"))
    cat = (tpl.get("category") or "").upper()
    if cat in ("NVIDIA", "AMD"):
        compat["category"] = cat
    allowed = []
    for v in tpl.get("allowedCudaVersions") or []:
        try:
            allowed.append(float(v))
        except (TypeError, ValueError):
            pass
    if allowed:
        compat["allowed_cuda"] = allowed
    compat["min_disk"] = tpl.get("disk") or 0
    compat["ports"] = tpl.get("ports") or []
    compat["env"] = tpl.get("env") or {}
    print(f"Template: {tpl.get('name') or template_ref}  image: {tpl.get('imageName')}")
    print(f"  compat: {describe_compat(compat)}")
    if compat.get("allowed_cuda"):
        print(f"  allowed CUDA versions: "
              f"{', '.join(f'{v:g}' for v in sorted(compat['allowed_cuda']))}")
    if compat["min_disk"]:
        print(f"  requires: container disk >= {compat['min_disk']} GB")
    return compat


def pick_gpu(max_price, cloud_type, wanted=None, compat=None, args_ref=None):
    """Return (gpu, price). If wanted is None, pick cheapest within cap
    (ties broken by more VRAM). Filters by template compatibility and
    args_ref.min_vram if given."""
    types = graphql(GPU_TYPES_QUERY)["gpuTypes"]
    if compat:
        types = [g for g in types if gpu_compatible(g, compat)]
        if not types:
            die("No GPU types compatible with this template image")
    if wanted is None and getattr(args_ref, "min_vram", 0):
        mv = args_ref.min_vram
        types = [g for g in types if (g.get("memoryInGb") or 0) >= mv]
        if not types:
            die(f"No compatible GPUs with >= {mv} GB VRAM within cap")
    if wanted:
        w = wanted.lower()
        for g in types:
            if g["id"].lower() == w or g["displayName"].lower() == w:
                if compat and not gpu_compatible(g, compat):
                    die(f"GPU {g['displayName']} is NOT compatible with the "
                        f"template image ({describe_compat(compat)})")
                price = gpu_price(g, cloud_type)
                if price is None or price > max_price:
                    die(f"GPU {g['displayName']} costs {money(price)}/hr — over cap {money(max_price)}/hr")
                return g, price
        die(f"GPU type '{wanted}' not found (or not compatible with template)")
    candidates = []
    for g in types:
        price = gpu_price(g, cloud_type)
        if price is not None and price <= max_price:
            candidates.append((price, g))
    if not candidates:
        die(f"No GPU types <= {money(max_price)}/hr on {cloud_type} cloud"
            + (" compatible with the template" if compat else ""))
    candidates.sort(key=lambda t: (t[0], -(t[1].get("memoryInGb") or 0)))
    return candidates[0]


# ----------------------------------------------------------------- pod setup

def build_pod_payload(args, gpu, price, compat=None):
    env = {}
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    disk = args.container_disk
    min_disk = (compat or {}).get("min_disk") or 0
    if disk < min_disk:
        print(f"  container disk bumped {disk} -> {min_disk} GB "
              f"(template requires >= {min_disk})")
        disk = min_disk
    payload = {
        "name": args.name,
        "gpuTypeIds": [gpu["id"]],
        "gpuCount": args.gpu_count,
        "cloudType": args.cloud,
        "containerDiskInGb": disk,
        "volumeInGb": args.volume,
        "supportPublicIp": True,
        "publicIpType": "dynamic",
    }
    if args.volume:
        payload["volumeMountPath"] = "/workspace"
    if args.template:
        payload["templateId"] = args.template
        # template carries its own ports/env/disk — send only explicit overrides
        if args.ports:
            payload["ports"] = args.ports
        if env:
            payload["env"] = env
    elif args.image:
        payload["image"] = args.image
        payload["ports"] = args.ports or DEFAULT_PORTS
        payload["env"] = env
    else:
        die("Provide --template <id> or --image <docker-image>")
    if args.cloud == "COMMUNITY":
        payload["bidPerGpu"] = price
    return payload


def wait_running(pod_id, timeout, interval=5):
    print(f"Waiting for pod {pod_id} to reach RUNNING (timeout {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = rest("GET", f"/pods/{pod_id}")
        runtime = pod.get("runtime") or {}
        status = pod.get("desiredStatus")
        print(f"  [{datetime.now(timezone.utc):%H:%M:%S}Z] "
              f"status={status} ports={len(runtime.get('ports') or [])}")
        if status == "RUNNING" and runtime.get("ports"):
            return pod
        time.sleep(interval)
    die(f"Pod {pod_id} did not reach RUNNING within {timeout}s "
        f"(it may still be billing — stop/terminate it manually)")


def ssh_target(pod):
    """Extract (host, port) for SSH from a pod object."""
    runtime = pod.get("runtime") or {}
    ssh = next((p for p in runtime.get("ports") or []
                if p.get("privatePort") == 22 and p.get("publicPort")), None)
    if not ssh:
        die("No public SSH port found — was 22/tcp in the pod's --ports?")
    host = pod.get("hostname") or ssh.get("ip")
    if not host:
        die("Could not determine pod hostname/IP")
    return host, int(ssh["publicPort"])


# ------------------------------------------------------------------ SSH layer

def ssh_connect(host, port, args):
    try:
        import paramiko
    except ImportError:
        die("'paramiko' is required for SSH operations: pip install paramiko")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = dict(hostname=host, port=port, username=SSH_USER,
                  timeout=30, banner_timeout=30, auth_timeout=30)
    if getattr(args, "password", None):
        kwargs["password"] = args.password
    else:
        key = getattr(args, "ssh_key", None) or os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa")
        if not Path(key).exists():
            die(f"SSH key not found: {key} (use --ssh-key or --password)")
        kwargs["key_filename"] = key
    client.connect(**kwargs)
    return client


def ssh_exec(client, command, timeout=600):
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


# --------------------------------------------------------------------- state

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_state():
    if not STATE_FILE.exists():
        die(f"No state file {STATE_FILE} — run 'create' first or pass a pod id")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def resolve_pod_id(ref):
    if ref and ref != "last":
        return ref
    return load_state()["pod_id"]


def fresh_pod(ref):
    pod_id = resolve_pod_id(ref)
    return pod_id, rest("GET", f"/pods/{pod_id}")


# ----------------------------------------------------------------- commands

def cmd_balance(args):
    b = get_balance()
    print(f"Balance:     {money(b['clientBalance'])}")
    print(f"Spend/hr:    {money(b['currentSpendPerHr'])}")
    print(f"Spend limit: {money(b['spendLimit'])}")


def cmd_gpus(args):
    compat = template_compat(args.template)
    types = [g for g in graphql(GPU_TYPES_QUERY)["gpuTypes"]
             if not compat or gpu_compatible(g, compat)]
    if args.min_vram:
        types = [g for g in types if (g.get("memoryInGb") or 0) >= args.min_vram]
        if not types:
            die(f"No GPUs with >= {args.min_vram} GB VRAM matching the filters")
    rows = []
    for g in types:
        price = gpu_price(g, args.cloud) if args.cloud != "ALL" \
            else (g.get("lowestPrice") or {}).get("uninterruptablePrice")
        rows.append((price if price is not None else float("inf"), g))
    rows.sort(key=lambda r: r[0])
    cap = args.max_price if args.max_price is not None else float("inf")
    print(f"{'GPU':<40} {'VRAM':>7}  {'price/hr':>8}  id")
    for price, g in rows:
        pstr = "n/a" if price == float("inf") else money(price)
        mark = "" if price <= cap else "  (over cap)"
        vram = f"{g.get('memoryInGb') or '?'}GB"
        print(f"{g['displayName']:<40} {vram:>7}  {pstr:>8}  {g['id']}{mark}")


def cmd_create(args):
    compat = template_compat(args.template)
    gpu, price = pick_gpu(args.max_price, args.cloud, args.gpu, compat, args)
    print(f"GPU: {gpu['displayName']} ({gpu.get('memoryInGb')} GB) @ {money(price)}/hr "
          f"(cap {money(args.max_price)}/hr)")
    payload = build_pod_payload(args, gpu, price, compat)
    pod = rest("POST", "/pods", payload)
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}  (cost/hr: {money(pod.get('costPerHour', price))})")
    save_state({
        "pod_id": pod_id,
        "name": args.name,
        "gpu": gpu["id"],
        "gpu_name": gpu["displayName"],
        "cost_per_hr": pod.get("costPerHour", price),
        "template": args.template,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    pod = wait_running(pod_id, args.wait_timeout)
    host, port = ssh_target(pod)
    state = load_state()
    state.update({"ssh_host": host, "ssh_port": port})
    save_state(state)
    print(f"SSH: ssh {SSH_USER}@{host} -p {port}")
    print(f"State saved to {STATE_FILE}")


def cmd_status(args):
    pod_id, pod = fresh_pod(args.pod)
    runtime = pod.get("runtime") or {}
    print(f"Pod {pod_id}: desiredStatus={pod.get('desiredStatus')}  "
          f"cost/hr={money(pod.get('costPerHour'))}")
    print(f"  uptime: {runtime.get('uptimeInMinutes', 0)} min")
    for p in runtime.get("ports") or []:
        print(f"  port {p.get('privatePort')} -> {p.get('publicPort')} ({p.get('type')})")


def cmd_exec(args):
    pod_id, pod = fresh_pod(args.pod)
    host, port = ssh_target(pod)
    client = ssh_connect(host, port, args)
    try:
        code, out, err = ssh_exec(client, args.command, timeout=args.timeout)
        sys.stdout.write(out)
        if out and not out.endswith("\n"):
            print()
        sys.stderr.write(err)
        sys.exit(code)
    finally:
        client.close()


def cmd_send(args):
    pod_id, pod = fresh_pod(args.pod)
    host, port = ssh_target(pod)
    client = ssh_connect(host, port, args)
    try:
        sftp = client.open_sftp()
        sftp.put(args.local, args.remote)
        sftp.close()
        print(f"Sent {args.local} -> {pod_id}:{args.remote}")
    finally:
        client.close()


def cmd_recv(args):
    pod_id, pod = fresh_pod(args.pod)
    host, port = ssh_target(pod)
    client = ssh_connect(host, port, args)
    try:
        sftp = client.open_sftp()
        sftp.get(args.remote, args.local)
        sftp.close()
        print(f"Received {pod_id}:{args.remote} -> {args.local}")
    finally:
        client.close()


def cmd_stop(args):
    pod_id = resolve_pod_id(args.pod)
    rest("POST", f"/pods/{pod_id}/stop")
    print(f"Pod {pod_id} stop requested.")


def cmd_start(args):
    pod_id = resolve_pod_id(args.pod)
    rest("POST", f"/pods/{pod_id}/start")
    print(f"Pod {pod_id} start requested.")


def cmd_terminate(args):
    pod_id = resolve_pod_id(args.pod)
    rest("DELETE", f"/pods/{pod_id}")
    print(f"Pod {pod_id} terminated.")
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print(f"Removed {STATE_FILE}")


def cmd_session(args):
    t0 = time.time()

    print("=== 1/7 Balance before ===")
    before = get_balance()
    print(f"  Balance: {money(before['clientBalance'])}")

    print("=== 2/7 GPU selection ===")
    compat = template_compat(args.template)
    gpu, price = pick_gpu(args.max_price, args.cloud, args.gpu, compat, args)
    print(f"  GPU: {gpu['displayName']} ({gpu.get('memoryInGb')} GB) @ {money(price)}/hr "
          f"(cap {money(args.max_price)}/hr)")

    print("=== 3/7 Create pod ===")
    payload = build_pod_payload(args, gpu, price, compat)
    pod = rest("POST", "/pods", payload)
    pod_id = pod["id"]
    print(f"  Pod {pod_id} created (cost/hr: {money(pod.get('costPerHour', price))})")
    save_state({"pod_id": pod_id, "name": args.name, "gpu": gpu["id"],
                "cost_per_hr": pod.get("costPerHour", price),
                "created_at": datetime.now(timezone.utc).isoformat()})

    print("=== 4/7 Deploy ===")
    pod = wait_running(pod_id, args.wait_timeout)
    host, port = ssh_target(pod)
    print(f"  SSH endpoint: {SSH_USER}@{host}:{port}")
    state = load_state()
    state.update({"ssh_host": host, "ssh_port": port})
    save_state(state)

    print("=== 5/7 SSH work ===")
    client = ssh_connect(host, port, args)
    try:
        sftp = client.open_sftp()
        for spec in args.send or []:
            local, _, remote = spec.partition(":")
            remote = remote or Path(local).name
            sftp.put(local, remote)
            print(f"  sent {local} -> {remote}")
        if args.script:
            remote_script = "/tmp/runpod_session_script.sh"
            sftp.put(args.script, remote_script)
            print(f"  uploaded script -> {remote_script}")
        sftp.close()

        if args.script:
            code, out, err = ssh_exec(client, f"bash {remote_script}", timeout=args.timeout)
            print(out)
            if err:
                sys.stderr.write(err)
            if code != 0:
                print(f"WARNING: script exited with code {code}", file=sys.stderr)
        for cmd in args.cmd or []:
            code, out, err = ssh_exec(client, cmd, timeout=args.timeout)
            print(f"$ {cmd}")
            print(out)
            if err:
                sys.stderr.write(err)

        sftp = client.open_sftp()
        for spec in args.recv or []:
            remote, _, local = spec.partition(":")
            local = local or Path(remote).name
            sftp.get(remote, local)
            print(f"  recv {remote} -> {local}")
        sftp.close()
    finally:
        client.close()

    print("=== 6/7 Stop & terminate ===")
    if not args.keep:
        rest("POST", f"/pods/{pod_id}/stop")
        print(f"  Pod {pod_id} stopped.")
    rest("DELETE", f"/pods/{pod_id}")
    print(f"  Pod {pod_id} terminated.")
    if STATE_FILE.exists() and not args.keep:
        STATE_FILE.unlink()

    print("=== 7/7 Session cost ===")
    time.sleep(5)  # balance updates can lag a few seconds after termination
    after = get_balance()
    cost = (before["clientBalance"] or 0) - (after["clientBalance"] or 0)
    wall_min = (time.time() - t0) / 60
    print(f"  Balance before: {money(before['clientBalance'])}")
    print(f"  Balance after:  {money(after['clientBalance'])}")
    print(f"  Session cost:   {money(cost)}  "
          f"(wall {wall_min:.1f} min, GPU rate {money(price)}/hr)")


# -------------------------------------------------------------------- parser

def add_pod_args(p):
    p.add_argument("--template", help="RunPod pod template id")
    p.add_argument("--image", help="Docker image (alternative to --template)")
    p.add_argument("--gpu", help="GPU type id or name (default: cheapest within cap)")
    p.add_argument("--max-price", type=float, default=2.0, help="Max $/hr (default 2.0)")
    p.add_argument("--min-vram", type=int, default=0,
                   help="Min GPU VRAM in GB (API doesn't expose template VRAM "
                        "requirements — set manually, e.g. --min-vram 24)")
    p.add_argument("--cloud", default="SECURE", choices=CLOUD_CHOICES)
    p.add_argument("--name", default=f"ephemeral-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--container-disk", type=int, default=20)
    p.add_argument("--volume", type=int, default=0, help="Persistent volume GB (mounted at /workspace)")
    p.add_argument("--ports", default=None,
                   help="Port spec (default: template's ports, or 22/tcp,8888/http)")
    p.add_argument("--env", action="append", metavar="K=V",
                   help="Env var for the pod, e.g. --env PUBLIC_KEY='ssh-ed25519 AAAA...'")
    p.add_argument("--wait-timeout", type=int, default=600)


def build_parser():
    ssh_opts = argparse.ArgumentParser(add_help=False)
    ssh_opts.add_argument("--ssh-key", help="SSH private key path (default ~/.ssh/id_rsa)")
    ssh_opts.add_argument("--password", help="SSH password (alternative to key)")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd_name", required=True)

    sub.add_parser("balance", help="Show balance / spend rate / limit")

    g = sub.add_parser("gpus", help="List GPU types and prices")
    g.add_argument("--max-price", type=float, default=None)
    g.add_argument("--cloud", default="SECURE", choices=CLOUD_CHOICES)
    g.add_argument("--template", help="Only show GPUs compatible with this template")
    g.add_argument("--min-vram", type=int, default=0, help="Min GPU VRAM in GB")

    c = sub.add_parser("create", help="Create + deploy a pod")
    add_pod_args(c)

    s = sub.add_parser("status", help="Show pod status")
    s.add_argument("pod", nargs="?", default="last")

    e = sub.add_parser("exec", parents=[ssh_opts], help="Run a command over SSH")
    e.add_argument("pod", nargs="?", default="last")
    e.add_argument("command")
    e.add_argument("--timeout", type=int, default=600)

    sd = sub.add_parser("send", parents=[ssh_opts], help="Upload a file (SFTP)")
    sd.add_argument("pod", nargs="?", default="last")
    sd.add_argument("local")
    sd.add_argument("remote")

    rv = sub.add_parser("recv", parents=[ssh_opts], help="Download a file (SFTP)")
    rv.add_argument("pod", nargs="?", default="last")
    rv.add_argument("remote")
    rv.add_argument("local")

    st = sub.add_parser("stop", help="Stop the pod")
    st.add_argument("pod", nargs="?", default="last")

    sta = sub.add_parser("start", help="Start a stopped pod")
    sta.add_argument("pod", nargs="?", default="last")

    t = sub.add_parser("terminate", help="Terminate (delete) the pod")
    t.add_argument("pod", nargs="?", default="last")

    ss = sub.add_parser("session", parents=[ssh_opts],
                        help="Full lifecycle: create -> work -> terminate -> cost")
    add_pod_args(ss)
    ss.add_argument("--send", action="append", metavar="LOCAL:REMOTE")
    ss.add_argument("--recv", action="append", metavar="REMOTE:LOCAL")
    ss.add_argument("--script", help="Local shell script to upload and run")
    ss.add_argument("--cmd", action="append", help="Inline command to run (repeatable)")
    ss.add_argument("--timeout", type=int, default=3600, help="Per-command timeout (s)")
    ss.add_argument("--keep", action="store_true", help="Stop but do not terminate")

    return p


COMMANDS = {
    "balance": cmd_balance,
    "gpus": cmd_gpus,
    "create": cmd_create,
    "status": cmd_status,
    "exec": cmd_exec,
    "send": cmd_send,
    "recv": cmd_recv,
    "stop": cmd_stop,
    "start": cmd_start,
    "terminate": cmd_terminate,
    "session": cmd_session,
}


def main():
    args = build_parser().parse_args()
    COMMANDS[args.cmd_name](args)


if __name__ == "__main__":
    main()
