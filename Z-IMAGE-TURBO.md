# Z-Image-Turbo on RunPod — Deployment Plan

> **Goal**: Run Tongyi-MAI/Z-Image-Turbo (6B param text-to-image model) on a RunPod GPU pod to generate photorealistic images.

---

## 1. Model Overview

- **Model**: Z-Image-Turbo by Tongyi-MAI (Alibaba)
- **Architecture**: Scalable Single-Stream DiT (S3-DiT), 6B parameters
- **Text Encoder**: Qwen3 (2.6B, 36 layers, 2560 hidden)
- **VAE**: AutoencoderKL (16 latent channels)
- **Pipeline**: `ZImagePipeline` (Diffusers)
- **Inference**: 8 NFE (num_inference_steps=9), guidance_scale=0.0
- **Resolution**: 512×512 up to 2048×2048 (any aspect ratio)
- **License**: Apache 2.0
- **HF**: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

### Model Weights Size

| Component | Size |
|---|---|
| Transformer (DiT) | 22.93 GB |
| Text Encoder (Qwen3) | 6.98 GB |
| VAE | 0.16 GB |
| **Total download** | **~30 GB** |
| **VRAM at bfloat16** | **~30 GB** |
| **VRAM with inference overhead** | **~34-36 GB** |

---

## 2. RunPod Pod Setup

### Minimum Hardware

| GPU | VRAM | Cost (Community) | Notes |
|---|---|---|---|
| A40 | 48 GB | ~$0.49/hr | ✅ Best value — fits easily with headroom |
| A100 (40GB) | 40 GB | ~$1.14/hr | ⚠️ Tight — may need CPU offloading |
| A100 (80GB) | 80 GB | ~$1.39/hr | ✅ Overkill, plenty of headroom |
| RTX A6000 | 48 GB | ~$0.76/hr | ✅ Works fine |

**Recommended**: A40 Community ($0.49/hr) — cheapest option that fits the model.

### Create Pod

1. Go to https://runpod.io → **Deploy** → **GPU Cloud**
2. Select **A40** (or A100 80GB if A40 unavailable)
3. **Template**: `RunPod PyTorch` (official, includes CUDA + PyTorch)
4. **Storage**: Set **Container Disk** to at least **50 GB** (model is ~30 GB download)
5. **Volume**: Mount `/workspace` to a Network Volume (persists between pod restarts) — recommended
6. **Expose HTTP Ports**: 8888 (for Jupyter/gradio if needed)
7. Click **Deploy**

### Verify GPU

```bash
nvidia-smi
# Should show A40 with 48GB VRAM
```

---

## 3. Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install diffusers from source (required for ZImagePipeline)
pip install git+https://github.com/huggingface/diffusers

# Install other dependencies
pip install transformers accelerate sentencepiece
```

> ⚠️ **Do NOT use `pip install diffusers`** — the PyPI version may not yet include `ZImagePipeline`. Install from source.

### Verify Install

```bash
python3 -c "from diffusers import ZImagePipeline; print('ZImagePipeline OK')"
```

---

## 4. Download Model

### Option A: Auto-download (first run)

The model downloads automatically on first `from_pretrained()` call. ~30 GB download, takes 5-15 min depending on bandwidth.

### Option B: Pre-download to /workspace (recommended)

```bash
pip install -U huggingface_hub

# Download to persistent workspace storage
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir /workspace/models/Z-Image-Turbo
```

> Set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads (uses Rust-based parallel downloader).

### Using Pre-downloaded Model

```python
pipe = ZImagePipeline.from_pretrained(
    "/workspace/models/Z-Image-Turbo",  # local path instead of HF repo ID
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
```

---

## 5. Run Inference

### Basic Generation Script

```python
#!/usr/bin/env python3
"""Z-Image-Turbo inference on RunPod A40"""

import torch
from diffusers import ZImagePipeline

# Load pipeline
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# Optional: Flash Attention 2 (faster on A40)
# pipe.transformer.set_attention_backend("flash")

# Optional: torch.compile for faster repeated inference
# First run will be slow due to compilation
# pipe.transformer.compile()

prompt = "Photorealistic image of an elderly man riding a sleek black motorcycle on a winding mountain road carved into a steep rock face. The man has a long grey beard flowing under his helmet, long silver-grey hair, and wears dark sunglasses. Towering mountain walls rise on one side, densely covered with lush hanging green plants, vines, and ferns draping over wet stone. On the other side, a rushing mountain river flows alongside the road, with multiple waterfalls cascading down the rocky cliffs into the river below. Mist from the waterfalls catches the light. The road curves dramatically through the gorge. Dramatic natural lighting, golden hour sun rays breaking through the mountain gap, wet road surface reflecting the sky. Shot on Sony A7R IV, 24mm wide angle, f/8, deep depth of field, ultra-realistic, National Geographic quality, 8K detail."

# Generate
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,   # = 8 DiT forwards (Turbo model)
    guidance_scale=0.0,       # MUST be 0 for Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("/workspace/z_image_output.png")
print(f"Saved to /workspace/z_image_output.png")
```

### Key Parameters

| Parameter | Turbo Value | Notes |
|---|---|---|
| `num_inference_steps` | 9 | Results in 8 DiT forward passes |
| `guidance_scale` | 0.0 | **Must be 0** for Turbo (no CFG) |
| `height` / `width` | 512–2048 | Any aspect ratio, total pixels flex |
| `torch_dtype` | `torch.bfloat16` | Best on modern GPUs |
| `generator` | `manual_seed(N)` | For reproducibility |

---

## 6. Performance Optimizations

### Flash Attention 2 (Recommended on A40)

```bash
pip install flash-attn --no-build-isolation
```

```python
pipe.transformer.set_attention_backend("flash")
```

### torch.compile (Faster After Warmup)

```python
# First inference takes ~2-5 min for compilation, then ~30-50% faster
pipe.transformer.compile()
```

### CPU Offloading (For VRAM < 40GB)

```python
# If VRAM is tight (e.g., A100 40GB), offload to CPU between steps
pipe.enable_model_cpu_offload()
```

### Quantization (For 16-24GB VRAM)

If running on a smaller GPU, you can use quantization to fit in ~16GB VRAM:

```python
from diffusers import ZImagePipeline, BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
# No .to("cuda") needed — quantized model auto-loads to GPU
```

---

## 7. Batch Generation Script

```python
#!/usr/bin/env python3
"""Batch image generation with Z-Image-Turbo"""

import torch
from diffusers import ZImagePipeline
import os

OUTPUT_DIR = "/workspace/z_image_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# Optional speedups
pipe.transformer.set_attention_backend("flash")

prompts = [
    "Photorealistic elderly man on black motorcycle, mountain gorge, waterfalls, lush green plants on rock walls, golden hour lighting",
    "Aerial view of a winding mountain road beside a rushing river, dramatic cliffs with hanging vegetation, misty waterfalls",
    "Close-up portrait of a bearded biker with sunglasses on a mountain road, waterfall in background, photorealistic 8K",
]

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(i),
    ).images[0]
    
    path = os.path.join(OUTPUT_DIR, f"output_{i:03d}.png")
    image.save(path)
    print(f"Saved: {path}")

print("Done! All images generated.")
```

---

## 8. Web UI (Gradio)

To generate images from a browser instead of SSH:

```python
#!/usr/bin/env python3
"""Gradio web UI for Z-Image-Turbo on RunPod"""

import torch
import gradio as gr
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")
pipe.transformer.set_attention_backend("flash")

def generate(prompt, seed, width, height):
    generator = torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Number(label="Seed", value=42),
        gr.Slider(512, 2048, step=128, value=1024, label="Width"),
        gr.Slider(512, 2048, step=128, value=1024, label="Height"),
    ],
    outputs=gr.Image(type="pil"),
    title="Z-Image-Turbo",
)

demo.launch(server_name="0.0.0.0", server_port=8888)
```

Access at: `https://<POD_ID>-8888.proxy.runpod.net`

---

## 9. Troubleshooting

### ❌ `mlx runner failed: failed to load MLX`

**Cause**: You installed the Ollama MLX version (`x/z-image-turbo:bf16`), which only runs on Apple Silicon.

**Fix**: Use the Diffusers pipeline instead (see Section 5). Remove the Ollama model:
```bash
ollama rm x/z-image-turbo:bf16
```

### ❌ `ImportError: cannot import name 'ZImagePipeline'`

**Cause**: PyPI `diffusers` package doesn't include ZImagePipeline yet.

**Fix**: Install from source:
```bash
pip install git+https://github.com/huggingface/diffusers
```

### ❌ `CUDA out of memory`

**Fixes** (try in order):
1. Use CPU offloading: `pipe.enable_model_cpu_offload()`
2. Use 4-bit quantization (Section 6)
3. Switch to larger GPU (A100 80GB)

### ❌ Slow first inference

**Normal**: First run involves CUDA kernel compilation and model loading. Subsequent runs are fast (~5-15 sec for 1024×1024 on A40).

### ❌ `flash-attn` build fails

```bash
pip install flash-attn --no-build-isolation
# If that fails, skip it — SDPA (default) works fine, just slightly slower
```

---

## 10. Cost Estimates

| GPU | VRAM | Cost/hr | Cost/image (est.) | Notes |
|---|---|---|---|---|
| A40 | 48 GB | $0.49 | ~$0.002 | Best value |
| A100 80GB | 80 GB | $1.39 | ~$0.004 | Overkill but fast |
| A6000 | 48 GB | $0.76 | ~$0.003 | Same VRAM as A40 |

Assuming ~15-30 sec per 1024×1024 image after model load.

---

## 11. Quick-Start Checklist

- [ ] Deploy RunPod pod with A40 GPU, 50GB+ disk, PyTorch template
- [ ] SSH into pod, verify `nvidia-smi` shows GPU
- [ ] `pip install git+https://github.com/huggingface/diffusers transformers accelerate sentencepiece`
- [ ] `pip install flash-attn --no-build-isolation` (optional, faster)
- [ ] Create and run generation script (Section 5)
- [ ] First run downloads ~30 GB model, then generates image
- [ ] Retrieve image: `scp root@<pod-ip>:/workspace/z_image_output.png .` or use RunPod file browser

---

## 12. Why NOT Ollama

The Ollama version (`x/z-image-turbo:bf16`) is packaged with **MLX** (Apple Metal) backend — it **only runs on Apple Silicon Macs**. The model itself is a standard Diffusers pipeline that runs on any CUDA GPU via PyTorch. Using the native Diffusers pipeline gives you:

- Full control over parameters (steps, CFG, seed, resolution)
- Flash Attention support
- Quantization options for smaller GPUs
- Batch generation
- Gradio web UI
- No 500 errors

---

*Plan created: 2026-06-20*
*Source: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo*