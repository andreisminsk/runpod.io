# Gemma 4 31B — Available Quantizations

> Last updated: July 2026

## Overview

Gemma 4 31B is a 31-billion parameter multimodal (image-text-to-text) model from Google. Multiple quantized versions are available on HuggingFace from both **Google** (official) and **Unsloth** (community). This document catalogs all available quantization variants, their file sizes, and trade-offs.

---

## Google Official Quantizations

Google provides a limited set of official quantized variants:

| Repo | Format | Quantization | Model Size | Notes |
|------|--------|-------------|-----------|-------|
| `google/gemma-4-31B-it-qat-q4_0-gguf` | GGUF | Q4_0 (QAT) | ~17.65 GB | Includes mmproj (~1.2 GB) |
| `google/gemma-4-31B-it-qat-w4a16-ct` | safetensors | W4A16 (compressed-tensors) | ~21.67 GB | 4-bit weights, 16-bit activations |
| `google/gemma-4-31B-it-qat-q4_0-unquantized` | safetensors | Unquantized (FP16/BF16) | ~65 GB (est.) | Full-precision QAT-trained weights |
| `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` | safetensors | Unquantized (FP16/BF16) | ~67 GB (est.) | Assistant-tuned variant |

**Key points:**
- Google only provides **one quantization level** (Q4_0) for the 31B model in GGUF format.
- The `w4a16-ct` variant uses the `compressed-tensors` format, suitable for vLLM / transformers.
- The `unquantized` variants contain full-precision weights that were QAT-trained (optimized for subsequent Q4_0 quantization).
- QAT (Quantization-Aware Training) means the model was trained with quantization in mind, so Q4_0 QAT typically preserves more quality than standard post-training Q4_0.

---

## Unsloth Community Quantizations

Repo: **`unsloth/gemma-4-31B-it-GGUF`**

Unsloth provides the widest range of quantization options, including both standard K-quants and their proprietary Unsloth Dynamic (UD) variants.

### Standard K-Quants

| Quantization | Size (GB) | Notes |
|---|---|---|
| Q3_K_S | 12.30 | Smallest standard |
| Q3_K_M | 13.72 | |
| Q4_0 | 16.15 | Same as Google's Q4_0 |
| Q4_K_S | 16.21 | |
| IQ4_NL | 16.10 | Importance-matrix 4-bit, non-linear |
| IQ4_XS | 15.25 | Importance-matrix 4-bit, extra small |
| Q4_1 | 17.81 | 4-bit with per-block scale |
| Q4_K_M | 17.07 | Popular quality/size balance |
| Q5_K_S | 19.68 | |
| Q5_K_M | 20.17 | |
| Q6_K | 23.47 | Near-lossless |
| Q8_0 | 30.39 | Essentially lossless |

### Unsloth Dynamic (UD) — Imatrix-optimized

| Quantization | Size (GB) | Notes |
|---|---|---|
| UD-IQ2_XXS | 7.95 | Smallest overall |
| UD-IQ2_M | 10.01 | |
| UD-IQ3_XXS | 11.02 | |
| UD-Q2_K_XL | 10.97 | |
| UD-Q3_K_XL | 14.31 | |
| UD-Q4_K_XL | 17.53 | Best quality/size trade-off |
| UD-Q5_K_XL | 20.39 | |
| UD-Q6_K_XL | 25.63 | |
| UD-Q8_K_XL | 32.60 | Highest quality |

### Additional Files

| File | Size | Purpose |
|---|---|---|
| mmproj-BF16.gguf | 1.12 GB | Multimodal projector (BF16) |
| mmproj-F16.gguf | 1.12 GB | Multimodal projector (F16) |
| mmproj-F32.gguf | 2.15 GB | Multimodal projector (F32) |
| mtp-gemma-4-31B-it.gguf | 0.48 GB | Multi-token prediction (speculative decoding) |
| imatrix_unsloth.gguf | 13 MB | Importance matrix calibration data |

---

## Unsloth QAT Quantization

Repo: **`unsloth/gemma-4-31B-it-qat-GGUF`**

A single QAT-optimized variant:

| Quantization | Size (GB) | Notes |
|---|---|---|
| UD-Q4_K_XL (QAT) | ~17.29 | QAT-trained, best Q4 quality |

Includes mmproj (BF16/F16/F32) and MTP draft model (~280 MB).

---

## Standard K-Quants vs. Unsloth Dynamic (UD)

### Standard K-Quants
These are the **original llama.cpp quantization schemes**. They use a fixed quantization strategy per tensor:

- **Q3_K** — 3-bit quantization with varying block sizes (S/M for different k-bit subsets)
- **Q4_0/Q4_1** — 4-bit with 32-element blocks; Q4_1 adds a per-block scale
- **Q4_K_S/Q4_K_M** — 4-bit K-quant with mixed precision (some layers get more bits than others)
- **Q5_K_S/Q5_K_M** — 5-bit, same mixed-precision approach
- **Q6_K** — 6-bit, near-lossless
- **Q8_0** — 8-bit, essentially lossless
- **IQ4_NL/IQ4_XS** — Importance-matrix 4-bit variants with simpler calibration

**Key trait:** Every layer gets the same quantization treatment (within the S/M variant). The bit budget is spread uniformly.

### Unsloth Dynamic (UD) — Imatrix-optimized
These combine two techniques:

1. **Importance Matrix (imatrix)** — Before quantizing, calibration data is run through the model to measure which layers and tensors matter most for output quality. The `imatrix_unsloth.gguf` file stores these importance scores.

2. **Dynamic bit allocation** — Instead of applying the same quantization to every layer, UD variants allocate more bits to important tensors and fewer bits to less important ones. For example, attention layers that heavily influence output might get Q5_K precision while less-critical MLP layers get Q3_K — all within the same model file.

### Comparison

| Aspect | Standard K-Quants | Unsloth Dynamic (UD) |
|--------|------------------|----------------------|
| **Bit allocation** | Uniform across layers | Weighted by importance |
| **Quality at same size** | Baseline | Typically better — important layers retain more precision |
| **Size at same quality** | Larger for equivalent quality | Smaller — bits are spent where they matter |
| **Calibration** | None or basic | Uses imatrix from real calibration data |
| **Naming** | Q4_K_M, Q5_K_S, etc. | UD-Q4_K_XL, UD-IQ2_M, etc. |
| **"XL" suffix** | N/A | Extra precision on critical tensors vs. the non-XL version |

**Bottom line:** UD variants give you better quality per GB than standard K-quants at the same approximate size. If choosing between a standard and UD variant of similar size, go with UD.

### Deep Dive: Why UD-Q8_K_XL Is Larger Than Q8_0

It seems counterintuitive that a "dynamic" quantization could be **larger** than a uniform 8-bit one. Here's why:

**Q8_0** gives every tensor exactly 8 bits. That's the ceiling — no layer gets more than 8 bits.

**UD-Q8_K_XL** doesn't redistribute the same bit budget. Instead, it **adds more bits to important tensors** — pushing some layers **above 8-bit** precision (e.g., into K-quant mixed-precision formats that use higher bit-widths for critical tensors). The "XL" suffix means extra precision on top of the base UD allocation.

The size increase comes from:

1. **Important layers get >8 bits** — Critical attention heads and output layers are quantized at higher precision (e.g., 12-bit or 16-bit K-quant blocks) because the imatrix identified them as disproportionately impacting output quality.
2. **Less important layers still get ~8 bits or slightly less** — But the savings there don't offset the extra bits on critical layers, because the important tensors tend to be larger.
3. **Net result: total bits > 8 bits × parameters** — The model is larger overall because the bit budget isn't fixed — it's expanded for quality-critical tensors.

#### Coding Task Analogy

Imagine you're a **senior developer** reviewing a codebase with limited "attention budget" (VRAM), deciding how much precision to devote to understanding each part.

**Q8_0 — Uniform Code Review**

You give **every file the same level of attention** — 8 minutes per file, whether it's:

- `main.py` — the core entry point (critical)
- `utils.py` — helper functions (important)
- `__init__.py` — just imports (trivial)
- `legacy_compat.py` — rarely called (unimportant)

Result: Trivial files get more scrutiny than they need; critical files don't get **extra** attention beyond the standard 8 minutes.

**UD-Q8_K_XL — Importance-Weighted Code Review**

Before reviewing, you run **profiling** (imatrix calibration) and discover:

| File | Q8_0 (min) | UD-Q8_K_XL (min) | Size (lines) |
|------|-----------|-------------------|-------------|
| `main.py` | 8 | **14** | 2,000 lines |
| `utils.py` | 8 | **10** | 800 lines |
| `__init__.py` | 8 | **3** | 50 lines |
| `legacy_compat.py` | 8 | **4** | 100 lines |

**Q8_0 total attention**: 8 × (2000 + 800 + 50 + 100) = 23,600 unit-minutes
**UD-Q8_K_XL total attention**: 14×2000 + 10×800 + 3×50 + 4×100 = **36,950 unit-minutes**

The extra time on the **large, critical files** (`main.py` getting 14 min instead of 8) far outweighs the savings on the **small, trivial files** (`__init__.py` dropping from 8 to 3). The net result is **more total attention** — which translates to a larger file size.

#### Why This Matters for Coding Tasks

When you ask the model to **write code**, it relies heavily on:

- **Attention layers** — understanding context, variable relationships, function call chains
- **Output projection layers** — generating syntactically correct tokens

These are exactly the layers that UD-Q8_K_XL **boosts above 8-bit**. So for coding:

- **Q8_0**: Every layer is 8-bit. Good enough for most tasks, essentially lossless.
- **UD-Q8_K_XL**: The layers that **matter most for code generation** get >8-bit precision. The difference may show up in:
  - Complex multi-step reasoning (nested logic, recursive algorithms)
  - Consistent variable naming across long contexts
  - Correct syntax in less common languages or frameworks

**Practical advice**: At 8-bit, both are near-lossless. The quality difference is negligible for most use cases. **Q8_0 is the better default** — it's smaller and the quality is already indistinguishable from full precision. Only choose UD-Q8_K_XL if you're chasing every last drop of quality and don't mind the extra ~2 GB.

### Inspection Results: UD-Q8_K_XL Per-Tensor Breakdown

Actual inspection of the GGUF file reveals that UD-Q8_K_XL uses **only two quantization types** — no intermediate K-quants:

| Type | Tensors | Size (GB) | % of Model |
|------|---------|-----------|------------|
| Q8_0 | 381 | 27.86 | 85.5% |
| F16 | 30 | 4.74 | 14.5% |
| F32 | 422 | 0.005 | ~0% (norms only) |

The F16 boost targets **6 out of 60 layers** — the "bookends" of the transformer:

| Layer | Type | Q | K | V | O | FFN_up | FFN_gate | FFN_down |
|-------|------|---|---|---|---|--------|----------|----------|
| 1 | sliding | **F16** | **F16** | Q8_0 | Q8_0 | **F16** | **F16** | **F16** |
| 52 | sliding | **F16** | **F16** | Q8_0 | Q8_0 | **F16** | **F16** | **F16** |
| 53 | full | **F16** | **F16** | — | Q8_0 | **F16** | **F16** | **F16** |
| 57 | sliding | **F16** | **F16** | Q8_0 | Q8_0 | **F16** | **F16** | **F16** |
| 58 | sliding | **F16** | **F16** | Q8_0 | Q8_0 | **F16** | **F16** | **F16** |
| 59 | full | **F16** | **F16** | — | Q8_0 | **F16** | **F16** | **F16** |
| All others | mixed | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |

**Key observations:**
- Layer 1 (early) and layers 52-59 (final 8) get F16 — initial feature extraction and output refinement
- V and O projections stay Q8_0 even in F16 layers — the imatrix found attention *scoring* (Q, K) and *knowledge transformation* (FFN) benefit more from 16-bit than attention *value aggregation* (V) and *output mixing* (O)
- The ~2 GB size premium comes entirely from these 30 F16 tensors (4.74 GB vs. ~2.37 GB if they were Q8_0)

**For coding tasks:** UD-Q8_K_XL is the better choice — F16 on Q/K attention scoring improves variable tracking across long contexts, and F16 on final-layer FFN improves syntax generation and reduces API hallucinations. But since 85.5% of the model is identical Q8_0, the practical difference is marginal. If the extra ~2 GB costs you context length or VRAM, Q8_0 is the smarter choice.

---

## VRAM Recommendations

| VRAM | Recommended Quantization | Notes |
|------|--------------------------|-------|
| 8 GB | UD-IQ2_XXS (7.95 GB) + mmproj-F16 | Tight fit, quality loss |
| 12 GB | Q3_K_S (12.3 GB) | Borderline; consider offloading |
| 16 GB | UD-Q4_K_XL (17.5 GB) + mmproj | Best quality/size trade-off |
| 24 GB | Q5_K_M (20.2 GB) or UD-Q5_K_XL (20.4 GB) + mmproj | Great quality |
| 32+ GB | Q6_K (23.5 GB) or Q8_0 (30.4 GB) | Near-original quality |

> **Note:** Add ~1.1 GB (mmproj-F16) to model size for total VRAM needed with vision support. Add ~0.5 GB for MTP speculative decoding if used. Context length (up to 262K tokens) also consumes additional VRAM at runtime.

---

## Ollama Availability

The official Ollama library (`ollama.com/library/gemma4`) provides limited 31B quantization options:

| Ollama Tag | Quantization | Size | Notes |
|---|---|---|---|
| `gemma4:31b` | Q4_K_M (default) | ~20 GB | Default pull — includes mmproj |
| `gemma4:31b-it-q4_K_M` | Q4_K_M (explicit) | ~20 GB | Same as default, explicit tag |

**To run:** `ollama run gemma4:31b`

### Using Other Quantizations on Ollama

For quantizations not in the official library (Q3, Q5, Q6, Q8, UD variants), pull directly from HuggingFace:

```bash
# Unsloth standard GGUF
ollama run hf.co/unsloth/gemma-4-31B-it-GGUF:Q5_K_M

# Unsloth Dynamic (UD)
ollama run hf.co/unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL

# Unsloth QAT
ollama run hf.co/unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL
```

> **Note:** When pulling from HuggingFace, the tag after the colon must match the exact filename suffix (e.g., `Q5_K_M`, `UD-Q3_K_XL`, `IQ4_XS`).

---

## Ollama & HuggingFace — Format Compatibility

Ollama **cannot** run every model on HuggingFace directly. Only **GGUF-format** models are supported.

### What works directly

Any GGUF model on HuggingFace can be pulled and run with:

```bash
ollama run hf.co/{username}/{repo}:{quantization}
```

Example: `ollama run hf.co/unsloth/gemma-4-31B-it-GGUF:Q5_K_M`

There are ~45K GGUF models on HuggingFace that work this way.

### What doesn't work directly

| Format | Direct Ollama support? | Workaround |
|--------|----------------------|------------|
| **GGUF** | ✅ Yes | — |
| **Safetensors** | ❌ No | Convert to GGUF first |
| **PyTorch `.bin`** | ❌ No | Convert to GGUF first |

Most models on HuggingFace are in **safetensors** format (the native HuggingFace format), which Ollama cannot import directly.

### Converting non-GGUF models for Ollama

1. **Download** the model from HuggingFace (safetensors / PyTorch)
2. **Convert** to GGUF using `llama.cpp` conversion scripts (e.g., `convert_hf_to_gguf.py`)
3. **Quantize** (optional) to reduce size
4. **Create a Modelfile** pointing to the GGUF file and run:
   ```bash
   ollama create mymodel -f Modelfile
   ollama run mymodel
   ```

> **Tip:** For fine-tuned adapters in safetensors format, you can create a Modelfile with a `FROM` command pointing to the base GGUF model and an `ADAPTER` command pointing to the adapter weights. See [Ollama Import Docs](https://docs.ollama.com/import).

---

## Links
>>>>>>> REPLACE`

- [Google Gemma 4 31B IT (base)](https://hf.co/google/gemma-4-31B-it)
- [Google QAT Q4_0 GGUF](https://hf.co/google/gemma-4-31B-it-qat-q4_0-gguf)
- [Google QAT W4A16 compressed-tensors](https://hf.co/google/gemma-4-31B-it-qat-w4a16-ct)
- [Google QAT unquantized](https://hf.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
- [Unsloth GGUF (all quantizations)](https://hf.co/unsloth/gemma-4-31B-it-GGUF)
- [Unsloth QAT GGUF](https://hf.co/unsloth/gemma-4-31B-it-qat-GGUF)
