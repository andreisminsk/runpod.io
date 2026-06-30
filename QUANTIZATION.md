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

## Links

- [Google Gemma 4 31B IT (base)](https://hf.co/google/gemma-4-31B-it)
- [Google QAT Q4_0 GGUF](https://hf.co/google/gemma-4-31B-it-qat-q4_0-gguf)
- [Google QAT W4A16 compressed-tensors](https://hf.co/google/gemma-4-31B-it-qat-w4a16-ct)
- [Google QAT unquantized](https://hf.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
- [Unsloth GGUF (all quantizations)](https://hf.co/unsloth/gemma-4-31B-it-GGUF)
- [Unsloth QAT GGUF](https://hf.co/unsloth/gemma-4-31B-it-qat-GGUF)
