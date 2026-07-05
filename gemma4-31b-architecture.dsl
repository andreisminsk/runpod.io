style=blueprint
title=Gemma 4 31B Architecture & Parameter Flow

*GEMMA4 "Gemma 4 31B (~31.25B)" -> EMBED, TRANSFORMER, VISION

EMBED "Embedding (1.41B)" | "262,144 × 5,376; token IDs → ℝ⁵³⁷⁶"

TRANSFORMER "60 Transformer Layers (29.29B)" -> SLIDE_LAYERS, FULL_LAYERS
TRANSFORMER | "Context: 256K tokens; all params are rank ≤ 2 tensors"

SLIDE_LAYERS "50 Sliding Attn (23.95B)" -> SLIDE_ATTN, SLIDE_FFN, SLIDE_NORMS
SLIDE_ATTN "Sliding Attention (~132M)" -> SLIDE_Q, SLIDE_K, SLIDE_V, SLIDE_O
SLIDE_Q "Q proj (44.0M)" | "5,376 × 8,192; ℝ⁵³⁷⁶ → ℝ⁸¹⁹²; 32 heads × 256 dim"
SLIDE_K "K proj (22.0M)" | "5,376 × 4,096; ℝ⁵³⁷⁶ → ℝ⁴⁰⁹⁶; 16 KV heads"
SLIDE_V "V proj (22.0M)" | "5,376 × 4,096; ℝ⁵³⁷⁶ → ℝ⁴⁰⁹⁶"
SLIDE_O "O proj (44.0M)" | "8,192 × 5,376; ℝ⁸¹⁹² → ℝ⁵³⁷⁶"
SLIDE_FFN "FFN (~346.8M)" -> SLIDE_UP, SLIDE_GATE, SLIDE_DOWN
SLIDE_UP "FFN up (115.6M)" | "5,376 × 21,504; ℝ⁵³⁷⁶ → ℝ²¹⁵⁰⁴; 4× expansion"
SLIDE_GATE "FFN gate (115.6M)" | "5,376 × 21,504; ℝ⁵³⁷⁶ → ℝ²¹⁵⁰⁴"
SLIDE_DOWN "FFN down (115.6M)" | "21,504 × 5,376; ℝ²¹⁵⁰⁴ → ℝ⁵³⁷⁶"
SLIDE_NORMS "Norms (~0.02M)" | "Sliding window: 1,024 tokens"

FULL_LAYERS "10 Full Attn (5.34B)" -> FULL_ATTN, FULL_FFN, FULL_NORMS
FULL_ATTN "Full Attention (~187.2M)" -> FULL_Q, FULL_K, FULL_O
FULL_Q "Q proj (88.1M)" | "5,376 × 16,384; ℝ⁵³⁷⁶ → ℝ¹⁶³⁸⁴; 32 heads × 512 dim"
FULL_K "K proj (11.0M)" | "5,376 × 2,048; ℝ⁵³⁷⁶ → ℝ²⁰⁴⁸; 4 KV heads"
FULL_O "O proj (88.1M)" | "16,384 × 5,376; ℝ¹⁶³⁸⁴ → ℝ⁵³⁷⁶"
FULL_FFN "FFN (~346.8M)" -> FULL_UP, FULL_GATE, FULL_DOWN
FULL_UP "FFN up (115.6M)" | "5,376 × 21,504; ℝ⁵³⁷⁶ → ℝ²¹⁵⁰⁴"
FULL_GATE "FFN gate (115.6M)" | "5,376 × 21,504; ℝ⁵³⁷⁶ → ℝ²¹⁵⁰⁴"
FULL_DOWN "FFN down (115.6M)" | "21,504 × 5,376; ℝ²¹⁵⁰⁴ → ℝ⁵³⁷⁶"
FULL_NORMS "Norms (~0.02M)"

VISION "Vision Encoder (0.55B)" | "mmproj; projects vision features → ℝ⁵³⁷⁶"

EMBED --> TRANSFORMER
VISION --> TRANSFORMER
