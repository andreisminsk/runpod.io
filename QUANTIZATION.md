# Q4 vs Q8 Quantization: The Real-World Quality Difference

## What the quantization levels actually mean

| Level | Bits per weight | Info retained | Model size (31B) |
|---|---|---|---|
| **Q4_K_M** | ~4.5 bits | ~85–90% of FP16 quality | ~18–20 GB |
| **Q8_0** | 8 bits | ~97–99% of FP16 quality | ~34 GB |

The jump from Q4 to Q8 is **not linear** — it follows diminishing returns. Most of the "loss" from quantization happens going from FP16 → Q4. Q8 recovers a significant chunk of that.

---

## Where you'll notice the difference

| Capability | Q4_K_M | Q8_0 | Difference noticeable? |
|---|---|---|---|
| **General knowledge / facts** | Solid | Slightly sharper | ⚠️ Marginal |
| **Reasoning / logic chains** | Occasional skips | More consistent | ✅ **Yes, especially on complex multi-step** |
| **Code generation** | Functional but sometimes imprecise | Cleaner, fewer bugs | ✅ **Yes, notably better** |
| **Creative writing** | Good, slight flatness | More nuanced vocabulary | ⚠️ Subtle |
| **Math / calculation** | More errors | Fewer errors | ✅ **Yes, measurable** |
| **Instruction following** | Occasionally misses nuances | More reliable | ⚠️ Moderate |
| **Long context coherence** | Slight drift over long outputs | More stable | ⚠️ Moderate |

---

## The honest summary

**For casual use** (chat, brainstorming, general Q&A): **Q4 is fine.** You may not notice the difference day-to-day. The model still "knows" the same things — it just occasionally fumbles details.

**For demanding use** (code, math, complex reasoning, long-form analysis): **Q8 is meaningfully better.** The difference shows up in:
- Fewer "dumb mistakes" on logic chains
- Better code that compiles/runs correctly the first time
- More consistent instruction following over long conversations
- Less hallucination on factual recall

## Rough quality scale

```
FP16  ████████████████████  100%  (reference)
Q8    ███████████████████░  ~97%  (near-indistinguishable from full)
Q5    ██████████████████░░  ~93%  (very good, slight edge cases)
Q4    ████████████████░░░░  ~87%  (good, noticeable gaps on hard tasks)
Q3    █████████████░░░░░░░  ~75%  (degraded, frequent errors)
```

---

## What this means for your Mac Mini decision

| If you're doing… | Minimum quantization | Mac Mini you need |
|---|---|---|
| Casual chat, brainstorming | Q4 is acceptable | M2 Pro 32GB ($900–$1,200 used) |
| Code, math, reasoning work | Q5 minimum, Q8 preferred | M4 Pro 48GB ($1,799 new) |
| Professional/production use | Q8 | M4 Pro 48GB ($1,799 new) |

**Bottom line:** The Q4→Q8 quality jump is **worth the extra $600–$900** if you're doing any serious reasoning or code work. If you're just chatting and exploring, Q4 on a cheaper M2 Pro 32GB is a perfectly viable budget option.
