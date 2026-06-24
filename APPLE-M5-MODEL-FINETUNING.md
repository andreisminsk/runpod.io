# Local LLMs & Fine-Tuning on Apple Silicon (64GB)

## 1. The 64GB Apple Silicon Landscape

Apple Silicon's unified memory architecture is a paradigm shift for local LLMs. The CPU and GPU share a 64GB memory pool, meaning there is no VRAM wall—unlike traditional NVIDIA cards where 24GB is the consumer ceiling. After OS overhead, expect ~56-58GB usable for models and KV cache.

### Model Sizing & Inference Capabilities

| Model (Params) | Quantization | Memory Needed | Runs on 64GB? | Est. Speed (tok/s) |
|---|---|---|---|---|
| **7-8B** | Q8 / FP16 | ~8-16GB | ✅ Easily | ~100-150 |
| **14B** | Q6-Q8 | ~10-14GB | ✅ Very comfortably | ~60-90 |
| **32-34B** | Q4_K_M | ~20-22GB | ✅ Comfortably | ~25-40 |
| **70B** | Q4_K_M | ~42-45GB | ✅ Fits (tight context) | ~10-20 |
| **120B+** | Any | 65GB+ | ❌ Too large | N/A |

**The Sweet Spot:** 64GB allows running a **70B Q4 model**—the threshold where models become genuinely useful for complex reasoning. Memory bandwidth is the primary bottleneck for token generation.

### Recommended Inference Stack
- **Ollama / LM Studio:** Easiest onboarding, OpenAI-compatible API.
- **llama.cpp:** Maximum performance, Metal-accelerated, most quantization options.
- **MLX:** Apple's native framework, ideal for fine-tuning and increasingly good for inference.

---

## 2. The Fine-Tuning Architecture: LoRA & QLoRA

Full fine-tuning updates every parameter, requiring hundreds of GBs of memory. **LoRA (Low-Rank Adaptation)** is the plugin/sidecar pattern: freeze the base model and train small, detachable adapter matrices. **QLoRA** adds base model 4-bit quantization (NF4) while keeping adapter training in higher precision (BF16/FP16).

### Memory Layout on 64GB

A 64GB Mac can comfortably QLoRA models up to 32B, and can *barely* squeeze in 70B QLoRA (requires small batch sizes and gradient checkpointing).

```
┌──────────────────────────────────────────────────────┐
│                    64GB Unified RAM                   │
├──────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌───────────────────────┐ │
│  │   Base Model (NF4)   │  │   LoRA Adapters (FP16)│ │
│  │   Frozen (4-bit)      │  │   Trainable (16-bit) │ │
│  │   e.g., 70B ≈ 38GB  │  │   e.g., r=16 ≈ 200MB │ │
│  └──────────────────────┘  └───────────────────────┘ │
│  ┌──────────────────────┐  ┌───────────────────────┐ │
│  │   Optimizer States    │  │   Activations / Ctx   │ │
│  │   (AdamW for Adapters)│  │   (Scratch space)     │ │
│  │   ≈ 400MB             │  │   ≈ 8-16GB            │ │
│  └──────────────────────┘  └───────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Recommended Training Stack
- **MLX-lm (Primary):** Native Apple Silicon, easiest setup, handles QLoRA natively without fighting CUDA abstractions.
- **Llama-Factory (Alternative):** More features, slightly more complex.
- **Data Prep:** Standard JSONL conversational formats.

### Architectural Smells & Risks in Fine-Tuning
1. **"Fine-tuning will fix bad prompts":** False. If RAG or prompt engineering solves it, fine-tuning is the wrong tool.
2. **Overfitting on tiny datasets:** Training 500 examples for 10 epochs causes memorization and brittleness. Use validation sets and early stopping.
3. **The Evaluation Gap:** Spending 90% of time training and 10% evaluating yields models that "sound better" but fail in production. Build eval datasets *before* training.
4. **LoRA Rank Too High:** Setting `r=64` or `r=128` defeats the purpose, risks catastrophic forgetting, and slows training. Start with `r=8` or `r=16`.

### Deployment: Merge vs. Runtime
Because 64GB lacks the memory to hold a base model + multiple adapters concurrently for serving, **Merge** the LoRA weights into the base model (using `mergekit` or MLX scripts) to create a single `.gguf` file. Serve this via Ollama or llama.cpp with zero inference overhead.

---

## 3. Strategic Use Cases: Behavior vs. Knowledge

**The Golden Rule:** Never fine-tune to add facts. Fine-tune to change behavior. Knowledge is a RAG problem; behavior is a fine-tuning problem.

| Problem | Solution | Analogy |
|---|---|---|
| "The model doesn't know about X" | **RAG** | Giving the employee a manual to read. |
| "The model won't output valid JSON" | **Fine-Tuning** | Training the employee on data-entry until it's muscle memory. |
| "The model is too slow/expensive" | **Distillation via Fine-Tuning** | Promoting the senior employee to manager, training a junior to do the 1 specific task. |
| "The model doesn't sound like us" | **Fine-Tuning** | Sending the employee to brand-voice bootcamp. |
| "The model uses the wrong reasoning steps" | **Fine-Tuning** | Teaching the employee a new decision-making framework. |

### The 5 High-ROI Use Cases for Fine-Tuning

1. **Format & Schema Compliance:** Baking strict JSON/XML/DSL output formatting into the model's default behavior, eliminating conversational preambles and structural hallucinations in automated pipelines.
2. **Task Distillation (Local Speed):** Using GPT-4 API to generate perfect outputs for a specific task, then fine-tuning a local 8B/14B model on those pairs. Result: 95% of GPT-4 quality at zero marginal cost and local latency.
3. **Tone, Style & Persona Alignment:** Internalizing a specific brand voice or persona so it doesn't "leak" back to the base model's default tone under long contexts or complex tasks.
4. **Domain-Specific Reasoning Patterns:** Teaching the model the *workflow* of a domain (e.g., the step-by-step triage process of a doctor), rather than just the medical facts.
5. **Tool/API Calling Optimization:** Training on synthetic API calls to push reliability from ~70% (with prompting) to ~99% reliable parameter passing and function selection.

---

## 4. Actionable Summary

For a 64GB M-series Mac, the optimal workflow is:
1. **Inference:** Run up to 70B Q4 models locally via Ollama/llama.cpp for private, unlimited API calls.
2. **Data Prep:** Format data as JSONL conversations targeting a specific behavior (format, style, or reasoning).
3. **Training:** Use MLX-lm for QLoRA (`r=16`, 1-3 epochs) on models up to 32B (comfortably) or 70B (with tight memory parameters).
4. **Evaluation:** Rigorously test against a holdout set before deploying.
5. **Deployment:** Merge the LoRA adapter into the base model and serve as a single `.gguf` file locally.