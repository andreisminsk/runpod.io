# GLM-5.2 — Model Analysis

## Overview

**GLM-5.2** is a flagship multimodal language model developed by **Zhipu AI** (Beijing Zhipu AI Technology Co., Ltd.), published under the `zai-org` organization on Hugging Face. It succeeds GLM-5.1 and is part of the **GLM (Generalized Language Model)** series, which includes predecessors like GLM-130B, GLM-4, and GLM-4.5.

- **Hugging Face Repo:** https://huggingface.co/zai-org/GLM-5.2
- **GitHub:** https://github.com/zai-org/GLM-5
- **Technical Report:** [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
- **License:** MIT (no regional restrictions)

---

## Architecture & Technical Specs

| Attribute | Value |
|---|---|
| **Architecture** | `glm_moe_dsa` (MoE + Dynamic Sparse Attention) |
| **Model Class** | `AutoModelForMultimodalLM` |
| **Total Parameters** | ~753B (753,329.9M) |
| **Context Window** | 1M tokens (stable long-horizon) |
| **Languages** | English, Chinese |
| **License** | MIT |
| **Library** | Transformers |
| **Task** | Text-generation (conversational) |
| **Downloads** | ~33.6K |
| **Likes** | 2,016 |

### Key Architectural Innovations

- **IndexShare:** Reuses the same indexer across every 4 sparse attention layers, reducing per-token FLOPs by **2.9×** at 1M context length.
- **MTP Layer:** Improved for speculative decoding, increasing acceptance length by up to **20%**.
- **Flexible Effort:** Multiple thinking effort levels to balance performance and latency for coding and reasoning tasks.
- **MoE (Mixture of Experts):** Only a fraction of the ~753B parameters are activated per inference, making it more efficient than a dense model of equivalent size.

---

## Repository File Structure

```
config.json              — Architecture hyperparameters (3,699 bytes)
generation_config.json   — Default generation parameters (194 bytes)
chat_template.jinja      — Chat formatting template (5,076 bytes)
model-000XX-of-00282.safetensors  — 282 weight shards (~5.36 GB each)
README.md                — Model card (7,606 bytes)
LICENSE                  — MIT license (1,065 bytes)
.gitattributes           — Git LFS config (1,570 bytes)
```

- **Sharding:** 282 safetensors files, each ~5.24–5.37 GB
- **Total Download Size:** ~1.51 TB

---

## Deployment Options

### 1. API (Easiest)
Use the [Z.ai API Platform](https://docs.z.ai/guides/llm/glm-5.2) — no local hardware needed.

### 2. Chat Interface
Available at [chat.z.ai](https://chat.z.ai).

### 3. Hosted Inference Providers
Available via:
- **Novita** (live)
- **Together** (live)
- **Fireworks AI** (live)

### 4. Local Deployment (Transformers)
- Requires ~1.5 TB storage for weights
- Needs significant GPU VRAM (multi-GPU setup mandatory for a 753B MoE model)
- Uses the `transformers` library with `AutoModelForMultimodalLM`
- Supports **Flexible Effort** — multiple thinking effort levels to balance performance vs. latency

```python
from transformers import AutoModelForMultimodalLM, AutoTokenizer

model = AutoModelForMultimodalLM.from_pretrained("zai-org/GLM-5.2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-5.2", trust_remote_code=True)
```

### 5. GitHub
Full source code at [github.com/zai-org/GLM-5](https://github.com/zai-org/GLM-5)

---

## Key Capabilities

- **Long-horizon tasks** — stable 1M-token context window
- **Advanced coding** with adjustable reasoning effort (Flexible Effort)
- **Multimodal** — text + vision (per the `AutoModelForMultimodalLM` model class)
- **Speculative decoding** support via MTP layers (20% improvement)
- **Bilingual** — strong English and Chinese support

---

## Benchmark Comparisons

GLM-5.2 is compared against GLM-5.1, Qwen3.7-Max, MiniMax M3, DeepSeek-V4-Pro, Claude Opus 4.8, GPT-5.5, and Gemini 3.1 Pro. See the [technical report](https://arxiv.org/abs/2602.15763) for full benchmark details.

---

## Predecessor Context (GLM-4.5)

For reference, the previous generation GLM-4.5 featured:
- 355B total parameters, 32B activated parameters (MoE)
- Trained on 23T tokens
- 70.1% on TAU-Bench, 91.0% on AIME 24, 64.2% on SWE-bench Verified
- Hybrid reasoning (thinking + direct response modes)

GLM-5.2 represents a significant scale-up and architectural evolution from GLM-4.5.

---

## Practical Takeaway

The main practical consideration: this is a **1.51 TB download** requiring a serious multi-GPU rig for local inference. For most use cases, the **API** or **hosted inference providers** (Novita/Together/Fireworks) will be far more practical than local deployment.

---

## RunPod Deployment Strategy

### The Core Challenge

GLM-5.2 is a 753B-parameter MoE model with ~40B active parameters per token. While MoE means fewer FLOPs per token, **all 753B weights must still reside in GPU memory**. This makes VRAM the primary constraint.

### Quantization Ladder (Unsloth Dynamic GGUF)

| Quant | Disk Size | Quality vs BF16 | Use Case |
|:---|:---|:---|:---|
| **UD-IQ2_M** | 239 GB | Noticeable degradation (~82% accuracy) | Budget / accessibility |
| **UD-Q3_K_XL** | 343 GB | Realistic floor for coding | Balanced cost-quality |
| **UD-Q4_K_XL** | 467 GB | Near-indistinguishable (sweet spot) | **Recommended** |
| **UD-Q6_K** | 626 GB | Visually lossless | High-fidelity needs |
| **BF16** | 1.51 TB | Full precision | Impractical on cloud GPUs |

> **Note:** Sizes above are weights only. Add 50–150 GB for KV cache depending on context length (1M context = massive KV cache). At shorter contexts (4K–32K), KV overhead is modest (~20–50 GB).

### RunPod GPU Pricing (Community Cloud, 2026)

| GPU | VRAM | Price/hr |
|:---|:---|:---|
| H100 94GB | 94 GB | $2.79 |
| H100 80GB | 80 GB | $2.39 |
| A100 80GB | 80 GB | $1.64 |
| L40 | 48 GB | $0.86 |
| A40 | 48 GB | $0.44 |
| RTX 6000 Ada | 48 GB | $0.76 |
| RTX 4090 | 24 GB | $0.69 |
| RTX 3090 | 24 GB | $0.43 |

### Deployment Tiers & Cost Estimates (2 Hours)

#### Tier 1: Recommended — Q4_K_XL on 8× H100 80GB

- **VRAM:** 8 × 80 = 640 GB (467 GB weights + ~170 GB KV headroom)
- **Quality:** Near-indistinguishable from BF16
- **Context:** Comfortable up to ~128K tokens; longer contexts possible with KV offloading
- **Framework:** vLLM with `--tensor-parallel-size 8`
- **Cost:** 8 × $2.39 = **$19.12/hr** → **$38.24 for 2 hours**

#### Tier 2: Balanced — Q3_K_XL on 6× H100 80GB

- **VRAM:** 6 × 80 = 480 GB (343 GB weights + ~137 GB KV headroom)
- **Quality:** Good for coding/agent tasks; slight degradation vs Q4
- **Context:** Comfortable up to ~64K tokens
- **Framework:** vLLM with `--tensor-parallel-size 6`
- **Cost:** 6 × $2.39 = **$14.34/hr** → **$28.68 for 2 hours**

#### Tier 3: Budget — UD-IQ2_M on 4× H100 80GB

- **VRAM:** 4 × 80 = 320 GB (239 GB weights + ~80 GB KV headroom)
- **Quality:** Noticeable degradation; acceptable for general chat
- **Context:** Comfortable up to ~32K tokens
- **Framework:** vLLM with `--tensor-parallel-size 4`
- **Cost:** 4 × $2.39 = **$9.56/hr** → **$19.12 for 2 hours**

#### Tier 4: A100 Alternative — Q4_K_XL on 8× A100 80GB

- **VRAM:** 8 × 80 = 640 GB (same as H100 tier)
- **Quality:** Same as Tier 1 (Q4_K_XL)
- **Performance:** ~30–40% slower than H100 (A100 lacks H100's FP8 support and has lower memory bandwidth)
- **Cost:** 8 × $1.64 = **$13.12/hr** → **$26.24 for 2 hours**

#### Tier 5: High-Fidelity — Q6_K on 8× H100 94GB

- **VRAM:** 8 × 94 = 752 GB (626 GB weights + ~126 GB KV headroom)
- **Quality:** Visually lossless
- **Context:** Comfortable up to ~64K tokens
- **Cost:** 8 × $2.79 = **$22.32/hr** → **$44.64 for 2 hours**

### Cost Summary Table

| Tier | Quant | GPUs | VRAM Total | Quality | $/hr | **2-Hour Cost** |
|:---|:---|:---|:---|:---|:---|:---|
| **1 (Recommended)** | Q4_K_XL | 8× H100 80GB | 640 GB | Sweet spot | $19.12 | **$38.24** |
| 2 (Balanced) | Q3_K_XL | 6× H100 80GB | 480 GB | Good | $14.34 | $28.68 |
| 3 (Budget) | IQ2_M | 4× H100 80GB | 320 GB | Acceptable | $9.56 | $19.12 |
| 4 (A100 Value) | Q4_K_XL | 8× A100 80GB | 640 GB | Sweet spot | $13.12 | $26.24 |
| 5 (High-Fidelity) | Q6_K | 8× H100 94GB | 752 GB | Lossless | $22.32 | $44.64 |

### Deployment Steps on RunPod

1. **Create a Pod** — Select multi-GPU configuration (e.g., 8× H100 80GB)
2. **Use the vLLM worker template** — RunPod provides a [vLLM worker](https://github.com/runpod-workers/worker-vllm) that simplifies deployment
3. **Set environment variables:**
   ```
   MODEL_NAME=zai-org/GLM-5.2
   QUANTIZATION=awq  (or gptq, depending on quantized variant)
   TENSOR_PARALLEL_SIZE=8
   MAX_MODEL_LEN=32768
   ```
4. **For GGUF quantization** — Use llama.cpp server instead of vLLM, with `--parallel 8` for tensor parallelism
5. **Storage** — Allocate at least 1.5 TB of network storage for the full model, or ~500 GB for Q4_K_XL
6. **Startup time** — Expect 10–20 minutes for model download and loading on first run

### Optimization Tips

- **Use Flexible Effort mode** — GLM-5.2 supports variable thinking effort; lower effort = faster inference = lower cost per token
- **Limit context length** — Set `MAX_MODEL_LEN` to your actual needs (e.g., 32K instead of 1M) to drastically reduce KV cache memory
- **Use speculative decoding** — The MTP layer gives ~20% acceptance length improvement; enable it in vLLM with `--speculative-decoding`
- **Spot instances** — RunPod Community Cloud offers lower prices but pods can be preempted; use for batch work, not production
- **Warm pool** — Keep a pod running if you need instant availability; otherwise accept cold-start latency

### When to Use API Instead

For 2 hours of usage, the cheapest RunPod option is **$19.12**. Compare this to:
- **Z.ai API** — Pay-per-token with no infrastructure management
- **Novita/Together/Fireworks** — Hosted inference with per-token pricing

If you don't need sustained throughput or custom fine-tuning, **API-based inference is likely more cost-effective** for intermittent use. RunPod shines when you need:
- Sustained high-throughput inference over many hours
- Custom fine-tuning or LoRA training
- Privacy (data never leaves your instance)
- Full control over inference parameters

---

## BF16 Full-Precision Deployment Plan on RunPod

### Why BF16?

BF16 (bfloat16) is the model's native precision. No quantization artifacts, no accuracy loss, no degraded reasoning. This is the deployment for when quality cannot be compromised — production inference, benchmarking, fine-tuning, or research requiring bit-exact outputs.

### The Brutal Math

| Factor | Value |
|:---|:---|
| Total parameters | 753,329,920,000 |
| Bytes per param (BF16) | 2 |
| **Weight memory (BF16)** | **~1,506 GB (1.51 TB)** |
| CUDA context overhead per GPU | ~5–8 GB |
| Activation memory per GPU | ~3–5 GB |
| KV cache (varies by context length) | See below |
| **Minimum VRAM just for weights** | **~1,506 GB** |

#### KV Cache Estimation

With 64 KV heads, head dimension 192, BF16:
- **Per token KV cost:** 2 × 64 × 192 × 2 bytes × num_layers ≈ ~49 KB per token (approximate, MoE layer count varies)
- **4K context:** ~0.2 GB per request
- **32K context:** ~1.6 GB per request
- **128K context:** ~6.3 GB per request
- **1M context:** ~50 GB per request

> **Bottom line:** BF16 deployment requires a **multi-node cluster**. No single RunPod node has enough VRAM.

---

### Hardware Configurations

#### Configuration A: 3-Node Cluster — 8× H100 80GB SXM per node (24 GPUs)

| Metric | Value |
|:---|:---|
| Total VRAM | 1,920 GB |
| Weights | ~1,506 GB |
| Per-GPU weight slice | ~62.75 GB |
| Per-GPU CUDA + activation overhead | ~10 GB |
| Per-GPU available for KV cache | ~7.25 GB |
| Total KV cache pool | ~174 GB |
| Max comfortable context | ~32K tokens (batch=1) or ~8K (batch=4) |
| Parallelism | TP=8, PP=3 |
| **GPU cost/hr** | 24 × $2.39 = **$57.36/hr** |
| **2-hour cost (GPU)** | **$114.72** |

**Verdict:** Tight but viable for short-context inference. KV cache is the bottleneck — you'll be limited on batch size and context length.

#### Configuration B: 3-Node Cluster — 8× H100 94GB per node (24 GPUs) ⭐ RECOMMENDED

| Metric | Value |
|:---|:---|
| Total VRAM | 2,256 GB |
| Weights | ~1,506 GB |
| Per-GPU weight slice | ~62.75 GB |
| Per-GPU CUDA + activation overhead | ~10 GB |
| Per-GPU available for KV cache | ~21.25 GB |
| Total KV cache pool | ~510 GB |
| Max comfortable context | ~128K tokens (batch=1) or ~32K (batch=4) |
| Parallelism | TP=8, PP=3 |
| **GPU cost/hr** | 24 × $2.79 = **$66.96/hr** |
| **2-hour cost (GPU)** | **$133.92** |

**Verdict:** Best balance of cost and capability. Comfortable KV headroom for moderate contexts and batched inference.

#### Configuration C: 4-Node Cluster — 8× H100 80GB SXM per node (32 GPUs)

| Metric | Value |
|:---|:---|
| Total VRAM | 2,560 GB |
| Weights | ~1,506 GB |
| Per-GPU weight slice | ~47.06 GB |
| Per-GPU CUDA + activation overhead | ~10 GB |
| Per-GPU available for KV cache | ~22.94 GB |
| Total KV cache pool | ~734 GB |
| Max comfortable context | ~256K tokens (batch=1) or ~64K (batch=4) |
| Parallelism | TP=8, PP=4 |
| **GPU cost/hr** | 32 × $2.39 = **$76.48/hr** |
| **2-hour cost (GPU)** | **$152.96** |

**Verdict:** Maximum KV headroom. Best for long-context workloads (128K+) or high-throughput batched serving.

#### Configuration D: 4-Node Cluster — 8× H100 94GB per node (32 GPUs) — MAXIMUM PERFORMANCE

| Metric | Value |
|:---|:---|
| Total VRAM | 3,008 GB |
| Weights | ~1,506 GB |
| Per-GPU weight slice | ~47.06 GB |
| Per-GPU CUDA + activation overhead | ~10 GB |
| Per-GPU available for KV cache | ~36.94 GB |
| Total KV cache pool | ~1,182 GB |
| Max comfortable context | ~512K+ tokens (batch=1) or ~128K (batch=4) |
| Parallelism | TP=8, PP=4 |
| **GPU cost/hr** | 32 × $2.79 = **$89.28/hr** |
| **2-hour cost (GPU)** | **$178.56** |

**Verdict:** Overkill for most use cases. Only justified if you need 1M-token context or massive batch throughput.

---

### Cost Summary — BF16 on RunPod (2 Hours)

| Config | GPUs | VRAM | Max Context | $/hr | **2-Hour Total** |
|:---|:---|:---|:---|:---|:---|
| A: 3×8× H100 80GB | 24 | 1,920 GB | ~32K | $57.36 | **$114.72** |
| **B: 3×8× H100 94GB** ⭐ | **24** | **2,256 GB** | **~128K** | **$66.96** | **$133.92** |
| C: 4×8× H100 80GB | 32 | 2,560 GB | ~256K | $76.48 | $152.96 |
| D: 4×8× H100 94GB | 32 | 3,008 GB | ~512K+ | $89.28 | $178.56 |

### Additional Costs

| Item | Cost |
|:---|:---|
| **Network Volume (1.5 TB)** | ~$0.10/GB/month = **~$150/month** (one-time model storage) |
| **Ephemeral NVMe per node** | Included in pod price (varies by config) |
| **Network egress** | **$0** (RunPod has free egress) |
| **Inter-node networking (InfiniBand/RoCE)** | Required for PP; included in cluster pricing |

---

### Step-by-Step Deployment Guide

#### Phase 1: Pre-Deployment (Before Launching Pods)

1. **Create a RunPod account** and add billing method (pay-as-you-go or credits)
2. **Create a Network Volume** (1.5 TB minimum) in the same region as your planned cluster
   - This persists the model across pod restarts — you only download once
   - Go to **Storage → Network Volumes → Create** → Select region → 1,500 GB
3. **Generate SSH keys** for inter-node communication
   ```bash
   ssh-keygen -t ed25519 -C "glm52-cluster" -f ~/.ssh/glm52_cluster
   ```
4. **Prepare a Docker image** with all dependencies:
   ```dockerfile
   FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

   RUN apt-get update && apt-get install -y \
       python3 python3-pip git openssh-server wget \
       && rm -rf /var/lib/apt/lists/*

   RUN pip install --no-cache-dir \
       torch==2.5.1 \
       transformers>=4.46.0 \
       vllm>=0.7.3 \
       ray[default]==2.42.0 \
       accelerate \
       sentencepiece \
       protobuf

   # Pre-download tokenizer
   RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('zai-org/GLM-5.2', trust_remote_code=True)"

   EXPOSE 8000 6379 8265
   ```
5. **Push the Docker image** to a registry (Docker Hub, GHCR, or RunPod's registry)

#### Phase 2: Cluster Setup

##### Step 1: Launch the Head Node

1. Go to **RunPod → Pods → Deploy**
2. Select **Cluster** mode (not single Pod)
3. Configure:
   - **GPU:** 8× H100 94GB (or 80GB)
   - **Image:** Your custom Docker image
   - **Network Volume:** Attach your 1.5 TB volume to `/workspace`
   - **Ports:** Expose 8000 (API), 8265 (Ray dashboard)
   - **SSH:** Add your public key
4. Launch the head node

##### Step 2: Download the Model

SSH into the head node and download the model to the network volume:

```bash
# This takes 30-90 minutes depending on network speed
# ~1.51 TB download from Hugging Face
huggingface-cli download zai-org/GLM-5.2 \
    --local-dir /workspace/models/glm-5.2-bf16 \
    --local-dir-use-symlinks False

# Verify download integrity
ls -la /workspace/models/glm-5.2-bf16/
# Should show 282 safetensors shards + config files
```

##### Step 3: Launch Worker Nodes

1. Launch 2 more pods (for 3-node cluster) with identical configuration
2. Ensure all nodes can reach each other on the internal network
3. Note each node's internal IP address

##### Step 4: Initialize Ray Cluster

On the **head node**:
```bash
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8
```

On each **worker node**:
```bash
ray start --address="<HEAD_NODE_IP>:6379" \
    --num-gpus=8
```

Verify cluster:
```bash
ray status
# Should show 3 nodes, 24 GPUs
```

##### Step 5: Launch vLLM with Multi-Node Serving

On the **head node**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/glm-5.2-bf16 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 3 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-prefix-caching \
    --disable-log-requests
```

Key parameters explained:
- `--tensor-parallel-size 8`: Each node's 8 GPUs share one model shard via NVLink
- `--pipeline-parallel-size 3`: Model split across 3 nodes via network
- `--max-model-len 32768`: Limit context to 32K (adjust based on config)
- `--max-num-seqs 4`: Limit concurrent requests (memory constraint)
- `--gpu-memory-utilization 0.92`: Use 92% of VRAM (leave headroom for CUDA)
- `--enable-prefix-caching`: Cache common prefixes for multi-turn conversations

##### Step 6: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/glm-5.2-bf16",
    "messages": [{"role": "user", "content": "Hello! What can you do?"}],
    "max_tokens": 256
  }'
```

#### Phase 3: Production Hardening

1. **Set up a reverse proxy** (nginx/caddy) with TLS termination
2. **Add authentication** — API key middleware in front of vLLM
3. **Configure autoscaling** — RunPod Serverless for burst traffic (if using API)
4. **Monitor GPU utilization** — Ray dashboard at `http://<head-ip>:8265`
5. **Set up logging** — Redirect vLLM logs to persistent storage on network volume
6. **Warm-up requests** — Send a few inference requests after loading to stabilize memory

#### Phase 4: Shutdown & Cost Control

```bash
# Stop vLLM gracefully
kill $(pgrep -f vllm)

# Stop Ray cluster
ray stop  # on all nodes

# Stop pods in RunPod dashboard (keep network volume to preserve model)
# Storage costs continue at ~$150/month for the 1.5TB volume
# Delete the network volume when no longer needed to stop storage charges
```

---

### Timing & Startup Estimates

| Phase | Duration | Notes |
|:---|:---|:---|
| Pod provisioning | 2–5 min | Multi-node clusters may take longer |
| Model download (first time) | 30–90 min | 1.51 TB from HuggingFace; depends on bandwidth |
| Model loading to GPU | 10–20 min | 1.51 TB from disk to 24 GPUs |
| Ray cluster init | 1–2 min | Head + 2 workers |
| vLLM startup + warmup | 5–10 min | Memory allocation, KV cache pre-allocation |
| **Total cold start** | **45–120 min** | First time; ~15–30 min on subsequent starts (model cached) |

> **Tip:** Keep the network volume persistent between sessions. After the first download, subsequent starts only need model loading time (~15–30 min).

---

### Performance Expectations (BF16, Config B: 3×8× H100 94GB)

| Metric | Estimated Value |
|:---|:---|
| Time to First Token (TTFT) | 1–3 seconds (short prompt) |
| Throughput (single request) | ~15–25 tokens/sec |
| Throughput (batch=4) | ~40–60 tokens/sec aggregate |
| Max context length | ~128K tokens |
| Concurrent requests | 2–4 (limited by KV cache) |
| Inter-node latency impact | ~10–20% vs single-node (pipeline parallel overhead) |

---

### Risk Mitigation

| Risk | Mitigation |
|:---|:---|
| **OOM during loading** | Reduce `--gpu-memory-utilization` to 0.88; reduce `--max-model-len` |
| **Inter-node network bottleneck** | Use RunPod clusters with NVLink/InfiniBand; avoid community cloud |
| **Model download failure** | Use `huggingface-cli download` with `--resume-download`; store on network volume |
| **Pod preemption (community cloud)** | Use Secure Cloud for production; community cloud for dev/testing |
| **Slow inference** | Enable `--enable-prefix-caching`; tune `--max-num-seqs`; consider speculative decoding with MTP |
| **Cold start too slow** | Keep 1 pod warm (costs ~$22/hr for 8× H100 94GB) or use RunPod Serverless with FlashBoot |

---

### Final Cost Breakdown — Config B (Recommended) for 2 Hours

| Item | Unit Cost | Quantity | Total |
|:---|:---|:---|:---|
| GPU compute (H100 94GB) | $2.79/hr | 24 GPUs × 2 hrs | $133.92 |
| Network volume (1.5 TB) | $0.10/GB/month | 1,500 GB × 1 month prorated | ~$0.21 |
| Network egress | $0 | — | $0.00 |
| **Total estimated cost** | | | **~$134.13** |

> **Note:** The network volume is a one-time setup cost (~$150/month). If you keep it between sessions, you only pay for the months it exists. The 2-hour GPU compute cost of **~$134** is the dominant expense.

### Comparison: BF16 vs Quantized (2-Hour Cost)

| Variant | Config | GPUs | Quality | 2-Hour Cost |
|:---|:---|:---|:---|:---|
| BF16 (full precision) | 3×8× H100 94GB | 24 | Perfect | **$133.92** |
| Q4_K_XL (4-bit) | 1×8× H100 80GB | 8 | Near-indistinguishable | $19.12 |
| Q3_K_XL (3-bit) | 1×6× H100 80GB | 6 | Good | $14.34 |
| IQ2_M (2-bit) | 1×4× H100 80GB | 4 | Acceptable | $9.56 |

> **Bottom line:** BF16 costs **7–14× more** than quantized variants. Only choose BF16 if you need bit-exact outputs, are fine-tuning, or are conducting research that cannot tolerate any quantization error. For production inference, Q4_K_XL on a single 8-GPU node at $19/2hrs is the practical choice.
