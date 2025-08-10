# Transformer Sidecar: Bolt On Persistent State Space Memory

> A tiny, external **fast‑weight memory** you can bolt onto any Hugging Face causal LM. It updates itself online (Hebbian‑style, gated by surprise/arousal), requires **no changes** to base model weights, and keeps a **constant‑size** footprint.

---

## Why this exists

Most “memory” approaches either **hoard everything** (vector DB / logs that grow forever) or **retrain the model** (adapters/LoRA). Sidecar takes a third path:

* **Selective**: only writes when the moment is novel, surprising, important, or explicitly rewarded.
* **Constant size**: two small, low‑rank matrices updated in place (no prompt bloat, no growing index).
* **Zero surgery**: the Transformer stays frozen; memory lives in a separate sidecar module.

In practice: it remembers the *right* things (preferences, warnings, changes) and leaves mundane chatter behind—like how humans do it.

---

## Key ideas (plain English)

* We keep two skinny matrices `U` and `V` (think: a tiny “notebook”).
* For each user turn we make a compact **key** from the model’s hidden state, read a **blend** of what’s relevant, and decide whether to **write a residual** (what’s new).
* A simple **gate** decides to write or not based on:

  * `S` = surprisal (NLL),
  * `N` = novelty (how different it is from memory),
  * `A` = arousal/importance (warning words, extreme numbers like **117°F**, notable places like **Death Valley**),
  * `C` = conflict with known facts (e.g., favorite color changed),
  * optional `R` = reward (user says “remember this”).
* When we **do** write, we Hebbian‑update `U,V` (no gradients, no backprop), and optionally flush short‑term **eligibility traces** to bind nearby turns.

---

## What makes it different from “just add a hard drive”?

**Hard drive / vector DB**

* Stores **everything**, grows forever
* Needs ANN search + prompt stuffing
* Latency and token costs grow with history

**Sidecar (this repo)**

* Stores **only what matters** (gated by S/N/A/C/R)
* **Constant‑size** memory (low‑rank `U,V`)
* Fast, tiny math ops; no prompt bloat

**Back‑of‑envelope** (fp32, example config `d_k=256`, `rank=64`):

* `U` = 256×64 = 16,384 floats ≈ **64 KB**
* `V` = same ≈ **64 KB**
* Self‑state (512 dims) ≈ **2 KB**
* **\~130 KB total**, regardless of conversation length

---

## Quick start

```bash
pip install torch transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sidecar_memory import SidecarV3
import torch

model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
mem = SidecarV3(model, tok, device=device)

# Normal generation with memory in the loop
print(mem.generate("Hello!", max_new_tokens=40))

# Teach a fact and recall it deterministically (KV + lexicon mask)
mem_text = "The user's favorite color is blue.\n"
ans = mem.generate_kv(mem_text, "Answer with just the color.", max_new_tokens=4,
                      stop_on_newline=False, allowed_words=["blue","red","green","yellow"]) 
print(ans)  # → "blue"

# Persist across sessions
mem.save_state("./demo_mem_state_v3")
# ... later ...
mem.load_state("./demo_mem_state_v3")
```

> Tip: for clean one‑word answers, prefer `generate_kv` with an `allowed_words` list.

---

## How it works (one‑page mental model)

* **Key creation**: project LM hidden state → key `k` (size `d_k`).
* **Read**: compute a mixing vector `a = softmax(U^T k / T)` and readout `r = V a`.
* **Residual**: `v_resid = norm(k - r)` captures what’s new.
* **Gate**: `G = wS*S + wN*N + wA*A + wC*C (+ wR*R)`; if `G > tau` → **commit**.
* **Commit**: decay + Hebbian rank‑1 updates to `U,V` with gain `eta0 * G`.
* **Eligibility**: hold recent `(k, v_resid)` for a few seconds; on the next commit, flush them too (binds multi‑turn events).
* **Steering**: a tiny GRU state produces a short `[STATE ...]` prefix to keep tone aligned.
* **Deterministic recall**: inject a short memory text into the KV cache, then greedily decode with an optional lexicon mask.

---

## Diagram (Mermaid)

```mermaid
flowchart LR
  subgraph Sidecar Memory (external)
    k[Key k from LM] --> UT[U^T k → a]
    UT --> a[mixture a]
    a --> Vr[V a → readout r]
    k --> diff[k − r → residual]
    diff --> commit{G > τ?}
    commit -- yes --> upd[Hebbian update of U,V]
    commit -- no --> elig[eligibility queue]
  end
  subgraph Gate
    S[NLL surprisal S]
    N[Novelty N]
    A[Arousal A]
    C[Conflict C]
    S --> sum
    N --> sum
    A --> sum
    C --> sum
    sum[G = wS*S + wN*N + wA*A + wC*C] --> commit
  end
  user[(User Prompt)] --> LM[Transformer LM]
  LM --> k
  Vr --> steer[State prefix]
  steer --> LM
```

---

## API (minimal)

```python
SidecarV3(
  model, tok, device="cpu",
  d_key=256, d_state=512, rank=64, key_topk=32,
  decay=0.995, eta0=0.1, temp=0.5,
  affect_weights=(0.3,0.15,0.5,0.05,0.0),
  fixed_tau=None, min_events=2, warmup_tau=0.90,
  s_mode='blend', s_alpha=0.6
)

.generate(prompt, max_new_tokens=64, temperature=0.7, top_p=0.9)
.generate_kv(mem_text, prompt, max_new_tokens=16,
             stop_on_newline=False, allowed_words=None)
.save_state(path)
.load_state(path)
```

**Important knobs**

* `affect_weights=(wS,wN,wA,wC,wR)` — balance surprisal/novelty/arousal/conflict/reward.
* `warmup_tau`, `min_events` — stricter early gating (avoid noisy commits at start).
* `rank`, `key_topk` — capacity and sparsity in key space.
* `eta0`, `decay`, `temp` — write gain, forgetting rate, read temperature.
* `s_mode={'blend','nll','novelty'}`, `s_alpha` — how to mix NLL vs novelty.

---

## Compatibility

* Works with Hugging Face **causal LMs** (e.g., GPT‑2/NeoX/Llama‑family) on CPU or CUDA.
* No model surgery; runs under `torch.no_grad()`.
* Safe with mixed precision; memory lives in the sidecar, not in base weights.

---

## FAQ

**Is this LoRA?** No. LoRA changes internal layer weights via gradient training. Sidecar is **external fast weights** updated online with Hebbian rules.

**Can I keep a full transcript elsewhere?** Yes—use a DB for audit/analytics. Sidecar is your **working memory**: small, selective, fast.

**What if the user changes their mind?** Conflicts raise `C` so new facts overwrite old ones; you can also call a future `forget()` utility or wipe state on disk.

**How big should `rank` be?** Start with 64 for small models; raise for broader domains or longer horizons.

---

## Roadmap

* Reward/RPE hook (`R`) for user‑marked importance
* Vision key‑path (CLIP/ViT) + visual arousal
* Consolidation/replay to compress traces
* Structured slots for preferences/tasks/identities

---

## Repository layout (suggested)

```
/sidecar_memory.py        # main module (v0.3.3)
/demo_notebook.ipynb      # quick demo & tests
/examples/                # tiny scripts
/docs/                    # this README, overview, tech details
```

---

## License

**MIT** — do what you want, just keep the notice.

---

## Acknowledgements

Thanks to open‑source HF, PyTorch, and the broader community exploring fast weights, associative memory, and human‑like gating.
