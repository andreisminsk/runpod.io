# Gemma 4 31B — Available Quantizations

> Last updated: July 2026

## Model Dimensions

| Dimension | Value | Math Description |
|-----------|-------|-----------------|
| Hidden size (d_model) | 5,376 | Each token is a vector in ℝ⁵³⁷⁶ |
| FFN intermediate size | 21,504 (4× expansion) | FFN projects tokens into ℝ²¹⁵⁰⁴ then back to ℝ⁵³⁷⁶ |
| Attention heads (Q) | 32 | Hidden vector split into 32 parallel subspaces |
| KV heads (sliding) | 16 | 16 key/value subspaces (Grouped Query Attention) |
| KV heads (full/global) | 4 | 4 global key/value subspaces (unified K/V) |
| Head dimension (sliding) | 256 | Each attention head operates in ℝ²⁵⁶ |
| Head dimension (full/global) | 512 | Each global head operates in ℝ⁵¹² |
| Layers | 60 (50 sliding + 10 full) | 60 sequential rank-2 tensor transformations |
| Vocabulary size | 262,144 | Embedding maps 262K token IDs → vectors in ℝ⁵³⁷⁶ |
| Context length | 262,144 (256K) | Maximum sequence length (axis size of the time dimension) |
| Sliding window | 1,024 | Local attention sees 1,024 tokens back (local ℝ¹⁰²⁴ window) |

### Per-Layer Parameter Counts

| Component | Sliding Attention Layer | Full Attention Layer | Math Description |
|-----------|------------------------|---------------------|-----------------|
| Q projection | 5,376 × 8,192 = 44.0M | 5,376 × 16,384 = 88.1M | Rank-2 tensor, linear map ℝ⁵³⁷⁶ → ℝ⁸¹⁹² (or ℝ¹⁶³⁸⁴) |
| K projection | 5,376 × 4,096 = 22.0M | 5,376 × 2,048 = 11.0M | Rank-2 tensor, linear map ℝ⁵³⁷⁶ → ℝ⁴⁰⁹⁶ (or ℝ²⁰⁴⁸) |
| V projection | 5,376 × 4,096 = 22.0M | *(unified with K)* | Rank-2 tensor, linear map ℝ⁵³⁷⁶ → ℝ⁴⁰⁹⁶ |
| O projection | 8,192 × 5,376 = 44.0M | 16,384 × 5,376 = 88.1M | Rank-2 tensor, linear map ℝ⁸¹⁹² → ℝ⁵³⁷⁶ (or ℝ¹⁶³⁸⁴ → ℝ⁵³⁷⁶) |
| FFN up | 5,376 × 21,504 = 115.6M | 5,376 × 21,504 = 115.6M | Rank-2 tensor, linear map ℝ⁵³⁷⁶ → ℝ²¹⁵⁰⁴ |
| FFN gate | 5,376 × 21,504 = 115.6M | 5,376 × 21,504 = 115.6M | Rank-2 tensor, linear map ℝ⁵³⁷⁶ → ℝ²¹⁵⁰⁴ |
| FFN down | 21,504 × 5,376 = 115.6M | 21,504 × 5,376 = 115.6M | Rank-2 tensor, linear map ℝ²¹⁵⁰⁴ → ℝ⁵³⁷⁶ |
| Norms + scales | ~0.02M | ~0.02M | Rank-1 tensors (vectors in ℝ⁵³⁷⁶) and rank-0 (scalars) |
| **Per-layer total** | **~479M** | **~534M** | |

### Total Parameter Count

| Component | Calculation | Parameters | Math Description |
|-----------|-------------|-----------|-----------------|
| Embedding (tied with output) | 262,144 × 5,376 | 1.41B | Rank-2 tensor, lookup table mapping token IDs → vectors in ℝ⁵³⁷⁶ |
| 50 sliding attention layers | 50 × 479M | 23.95B | 50 sequential rank-2 transformations per component |
| 10 full attention layers | 10 × 534M | 5.34B | 10 sequential rank-2 transformations per component |
| **Text model total** | | **~30.7B** | ~30.7B individual scalars (rank-0 tensors) |
| Vision encoder (mmproj) | | ~0.55B | Projects vision features into ℝ⁵³⁷⁶ |
| **Total with vision** | | **~31.25B** | |

### Tensor Rank Summary

All learnable parameters in this model are rank ≤ 2:

| Rank | Shape | What it represents | Example |
|------|-------|--------------------|---------|
| 0 (scalar) | `[1]` | Single scale value | `layer_output_scale.weight` |
| 1 (vector) | `[5376]` | Per-channel scale/normalization | `attn_norm.weight`, `ffn_norm.weight` |
| 2 (matrix) | `[5376, 8192]` | Linear transformation between vector spaces | `attn_q.weight`, `ffn_up.weight` |

**No rank-3 or higher tensors exist in this model.** The largest matrix is the embedding at `[262,144 × 5,376]` containing 1.41B individual scalars. The maximum axis size (ML "dimension") is 262,144 (vocabulary size).

> **Note:** Full attention layers use unified K/V projections (no separate V tensor) and 512-dim heads with 4 global KV heads, making Q and O projections 2× larger than sliding layers. This is why full attention layers have ~55M more parameters each.

---

## Notation Guide: ℝⁿ

Throughout this document, **ℝⁿ** (read: "R-n" or "real n-space") denotes the set of all vectors with **n** real-number components.

- **ℝ** = the set of all real numbers (…, -1.5, 0, 3.14, …)
- **ℝ³** = all possible 3D vectors: `[x, y, z]` where x, y, z are real numbers
- **ℝ⁵³⁷⁶** = all possible vectors with 5,376 real-number components

When we say a token is "a vector in ℝ⁵³⁷⁶", it means each token is represented by a list of 5,376 real numbers. A weight matrix that maps ℝ⁵³⁷⁶ → ℝ⁸¹⁹² takes a 5,376-number vector as input and produces an 8,192-number vector as output — it's a function between two vector spaces.

| Notation | Meaning | In this model |
|----------|---------|---------------|
| ℝ⁵³⁷⁶ | Vectors with 5,376 components | Token hidden states |
| ℝ⁸¹⁹² | Vectors with 8,192 components | Query (Q) output space |
| ℝ²¹⁵⁰⁴ | Vectors with 21,504 components | FFN intermediate space |
| ℝ²⁶²¹⁴⁴ | Vectors with 262,144 components | Vocabulary (token ID) space |

The superscript is the **dimension of the vector space** — how many numbers each vector contains. This is the math meaning of "dimension" (size of the space), not the tensor rank (number of axes).

### Origin of the ℝⁿ Notation

| Period | Development |
|--------|-------------|
| 1637 | Descartes invents Cartesian coordinates — the idea of ℝ² as a plane |
| 1888 | Peano formally defines vector spaces and ℝⁿ |
| 1930s–50s | Bourbaki group (French mathematicians) standardizes ℝⁿ notation in their textbooks |
| 1960s | Blackboard bold ℝ (double-struck) appears on university chalkboards to distinguish the set ℝ from the variable R |
| 1990s | LaTeX adopts `\mathbb{R}` for the symbol |

- **ℝ** = the set of real numbers, from the German **"Reell"** (real). The double-struck style distinguishes it from a variable named R.
- **The superscript ⁿ** = Cartesian product. ℝ² = ℝ × ℝ (all pairs), ℝ³ = ℝ × ℝ × ℝ (all triples), ℝⁿ = ℝ × ℝ × ... × ℝ (n times).

---

## ML vs. Math Terminology

The ML terminology is confusing because it uses "dimension" in a way that doesn't directly map to the math you learned.

### The Math You Know

| Rank | Name | Example | How many indices to address one element |
|------|------|---------|----------------------------------------|
| 0 | Scalar | `5` | None — it's just a number |
| 1 | Vector | `[1, 3, 7]` | One: `v[2] = 7` |
| 2 | Matrix | `[[1,2],[3,4]]` | Two: `m[1][0] = 3` |
| 3+ | Tensor | 3D grid of numbers | Three+: `t[2][1][0]` |

This is **tensor rank** (or order) — how many axes a data structure has.

### What ML Calls "Dimensions"

When ML people say "hidden dimension = 5,376", they're **not** talking about rank. They're talking about the **size along one axis** of a rank-1 or rank-2 tensor.

In math terms:

| ML term | What it actually is | Math description |
|---------|-------------------|-----------------|
| "Hidden dimension" = 5,376 | Size of a vector | A vector in **ℝ⁵³⁷⁶** — a 1D tensor with 5,376 components |
| "Q projection" = 5,376 × 8,192 | A matrix | A rank-2 tensor, a linear map **ℝ⁵³⁷⁶ → ℝ⁸¹⁹²** |
| "Embedding" = 262,144 × 5,376 | A matrix | A rank-2 tensor, a lookup table mapping token IDs to vectors in **ℝ⁵³⁷⁶** |

### The Key Confusion

In math, a vector `[1, 3, 7]` is **1-dimensional** (rank 1) but lives in **ℝ³** (a 3-dimensional space). These are two different uses of "dimension":

- **Rank** = how many axes the tensor has (math dimension)
- **Size** = how many elements along one axis (ML "dimension")

ML people say "dimension" when they mean "size along an axis." A weight matrix of shape `[5376, 8192]` is:

- **Rank 2** in math (it's a matrix)
- Has **dimensions** 5,376 and 8,192 in ML speak (sizes of its two axes)
- Contains **5,376 × 8,192 = 44M parameters** (individual scalars)

### The Full Mapping

| Math concept | Math term | ML term | Gemma 4 31B example |
|-------------|-----------|---------|---------------------|
| Single number | Scalar (rank 0) | Parameter / weight | `0.137` |
| Ordered list of numbers | Vector (rank 1) | "Dimension" = axis size | Token embedding: 5,376 numbers per token |
| Grid of numbers | Matrix (rank 2) | Weight matrix | Q projection: 5,376 × 8,192 numbers |
| 3D+ grid of numbers | Tensor (rank 3+) | Weight tensor | Not used in this model |

| ML phrase | What it actually means | Math description |
|-----------|----------------------|-----------------|
| "Hidden dimension = 5,376" | Size of the token vector | A vector in ℝ⁵³⁷⁶ |
| "Q projection = 5,376 × 8,192" | A weight matrix | A linear map ℝ⁵³⁷⁶ → ℝ⁸¹⁹² |
| "Embedding = 262,144 × 5,376" | A lookup table | Maps token IDs to vectors in ℝ⁵³⁷⁶ |
| "Parameters = 44M" | Number of scalars in a weight matrix | 5,376 × 8,192 = 44,032,512 individual numbers |

### Why It Matters

- **Same dimensions → same parameter count** (architecture is fixed)
- **Same parameter count → different file sizes** depending on quantization:
  - FP16: 44M params × 2 bytes = 88 MB
  - Q8: 44M params × 1 byte = 44 MB
  - Q4: 44M params × 0.5 bytes = 22 MB

The dimensions don't change with quantization — only the **precision of each parameter** changes. That's why Q8_0 and UD-Q8_K_XL have the same architecture (same dimensions, same parameter count of ~30.7B) but different file sizes (30.4 GB vs 32.6 GB) — some parameters are stored at higher precision.

---

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
