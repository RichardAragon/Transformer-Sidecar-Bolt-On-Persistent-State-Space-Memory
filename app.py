# Sidecar Memory — v0.3.3 (Affect/Arousal‑Gated Fast Weights)
# Add a human‑like affect channel (arousal) to surprise gating, plus simple conflict detection.
#  • G = wS·S (NLL surprisal) + wN·N (novelty) + wA·A (affect arousal) + wC·C (conflict) + wR·R (reward)
#  • Tiny AffectScorer: hazard lexicon + numeric outliers (e.g., 117°F) + named‑entity priors + punctuation/intensifiers
#  • Strict gate with cold‑start guard; residual fast‑weights; KV recall with lexicon mask

from textwrap import dedent

# ==============================
# 1) sidecar_memory.py (v0.3.3)
# ==============================
sidecar_code = dedent(r'''
from __future__ import annotations
import os, json, time, math, re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "0.3.3"

# ---------------- device helpers ----------------

def mod_device(mod: nn.Module):
    try:
        return next(mod.parameters()).device
    except StopIteration:
        for _, b in mod.named_buffers():
            return b.device
        return torch.device('cpu')

def to_mod(x: torch.Tensor, mod: nn.Module):
    return x.to(mod_device(mod))

# ---------------- utils ----------------

def _get_base_module(model: nn.Module):
    return getattr(model, 'transformer', None) or getattr(model, 'base_model', None) or model

def l2_normalize(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (an * bn).sum(dim=-1)

@torch.no_grad()
def sparsify_topk(x: torch.Tensor, top_k: int):
    if top_k <= 0 or top_k >= x.shape[-1]:
        return x
    vals = x.abs()
    thr = torch.topk(vals, k=top_k, dim=-1).values[..., -1:]
    mask = vals >= thr
    return x * mask

# ---------------- self‑state ----------------
class SelfStateManager(nn.Module):
    def __init__(self, d_state: int = 512, d_in: int = 256):
        super().__init__()
        self.d_state = d_state
        self.cell = nn.GRUCell(d_in, d_state)
        self.ln = nn.LayerNorm(d_state)
        self.register_buffer("state", torch.zeros(1, d_state))
    @torch.no_grad()
    def reset(self):
        self.state.zero_()
    @torch.no_grad()
    def get(self):
        return self.state.to(mod_device(self))
    def forward(self, x: torch.Tensor):
        s = self.cell(to_mod(x, self), to_mod(self.state, self))
        s = self.ln(s)
        if self.state.device != s.device:
            self.state.data = s.detach().clone()
        else:
            self.state.copy_(s)
        return self.state

# ---------------- fast weights (low‑rank) ----------------
class FastWeightsMemory(nn.Module):
    """Low‑rank fast weights A ≈ U V^T with Hebbian updates, decay, and sparse keys.
    Read: r = V @ softmax(U^T k / T)
    Commit: U,V ← λ U,V + η * (k ⊗ α^T, v_resid ⊗ α^T)
    """
    def __init__(self, d: int = 256, rank: int = 64, decay: float = 0.995, eta0: float = 0.1, temp: float = 0.5, key_topk: int = 32):
        super().__init__()
        self.d, self.r = d, rank
        self.decay = decay
        self.eta0 = eta0
        self.temp = temp
        self.key_topk = key_topk
        self.register_buffer('U', torch.zeros(d, rank))
        self.register_buffer('V', torch.zeros(d, rank))
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
    @torch.no_grad()
    def _alpha(self, k: torch.Tensor):
        k = sparsify_topk(k, self.key_topk)
        logits = (self.U.t() @ k.t()).squeeze(-1) / max(1e-6, self.temp)
        return torch.softmax(logits, dim=-1).unsqueeze(0)
    @torch.no_grad()
    def read(self, k: torch.Tensor) -> torch.Tensor:
        a = self._alpha(k)
        return a @ self.V.t()
    @torch.no_grad()
    def commit(self, k: torch.Tensor, v_resid: torch.Tensor, gain: float = 1.0):
        self.U.mul_(self.decay); self.V.mul_(self.decay)
        a = self._alpha(k)
        eta = self.eta0 * float(gain)
        self.U.add_(k.t() @ a)
        self.V.add_(v_resid.t() @ a)
        self.U.mul_(1.0 + (eta - 1.0)); self.V.mul_(1.0 + (eta - 1.0))
        self.U.div_(self.U.norm(dim=0, keepdim=True) + 1e-6)
        self.V.div_(self.V.norm(dim=0, keepdim=True) + 1e-6)

# ---------------- eligibility traces ----------------
@dataclass
class Elig:
    t: float
    k: torch.Tensor
    v: torch.Tensor
    s: float

class EligBuffer:
    def __init__(self, half_life_s: float = 10.0, window_s: float = 30.0):
        self.h = half_life_s; self.w = window_s; self.buf: List[Elig] = []
    def add(self, k: torch.Tensor, v: torch.Tensor, s: float):
        self.buf.append(Elig(time.time(), k.detach().cpu(), v.detach().cpu(), float(s)))
    def pop_alive(self) -> List[Elig]:
        now = time.time(); keep, out = [], []
        for e in self.buf:
            if now - e.t <= self.w: out.append(e)
            else: keep.append(e)
        self.buf = keep; return out

# ---------------- affect scorer (arousal) ----------------
class AffectScorer:
    def __init__(self):
        self.hazard = set('warning flashed emergency accident police danger crash burned died death hospital icu earthquake tornado hurricane wildfire explosion evacuate flood storm alarm heatwave heatstroke blackout fire rescue stranded'.split())
        self.intens = set('very extremely unbelievably incredibly super really so highly severely dangerously insanely'.split())
        self.ne_prior = ['death valley','icu','er','emergency room','police','court','earthquake','tornado','hurricane','wildfire','war','explosion']
        self.word_re = re.compile(r"[A-Za-z']+")
        self.num_unit_re = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*(°?\s*[Ff]|°?\s*[Cc]|mph|mi|miles|km|hours?|mins?|\$)?")
    def _lex_arousal(self, text: str) -> float:
        w = [t.lower() for t in self.word_re.findall(text)]
        haz = sum(1 for t in w if t in self.hazard)
        itf = sum(1 for t in w if t in self.intens)
        caps = sum(1 for t in re.findall(r"\b[A-Z]{3,}\b", text))
        excl = text.count('!')
        val = 0.15*min(3,haz) + 0.1*min(3,itf) + 0.05*min(3,caps) + 0.05*min(3,excl)
        return max(0.0, min(1.0, val))
    def _ne_arousal(self, text: str) -> float:
        tl = text.lower()
        hit = any(ne in tl for ne in self.ne_prior)
        return 0.25 if hit else 0.0
    def _num_arousal(self, text: str) -> float:
        a = 0.0
        for m in self.num_unit_re.finditer(text):
            if not m.group(1):
                continue
            x = float(m.group(1)); unit = (m.group(2) or '').lower().replace(' ', '')
            def sig(z):
                return 1/(1+math.exp(-z))
            if unit in ['°f','f','°f'] or 'f'==unit:
                a = max(a, sig((x-95)/5))
            elif unit in ['°c','c','°c'] or 'c'==unit:
                a = max(a, sig((x-35)/3))
            elif unit == 'mph':
                a = max(a, sig((x-80)/10))
            elif unit in ['mi','miles','km']:
                a = max(a, sig((x-300)/80))
            elif unit in ['hour','hours','mins','min']:
                a = max(a, sig((x-8)/2))
            elif unit == '$' or '$' in m.group(0):
                a = max(a, sig((math.log10(max(1.0,x)) - 3)/0.6))
            else:
                # generic z vs recent history could go here; keep simple
                pass
        return max(0.0, min(1.0, a))
    def arousal(self, text: str) -> float:
        a = self._lex_arousal(text) + self._ne_arousal(text) + self._num_arousal(text)
        return max(0.0, min(1.0, a))

# ---------------- surprise gate ----------------
class SurpriseGate:
    def __init__(self, wS=0.3, wN=0.15, wA=0.5, wC=0.05, wR=0.0, ema_beta=0.9, ksigma=0.5, fixed_tau: Optional[float]=None, min_events: int = 2, warmup_tau: float = 0.90):
        self.wS, self.wN, self.wA, self.wC, self.wR = wS, wN, wA, wC, wR
        self.beta = ema_beta; self.ksigma = ksigma
        self.fixed_tau = fixed_tau; self.mu = 0.3; self.var = 0.02
        self.count = 0; self.min_events = int(min_events); self.warmup_tau = float(warmup_tau)
    def score(self, S: float, N: float, A: float, C: float = 0.0, R: float = 0.0) -> float:
        return self.wS*S + self.wN*N + self.wA*A + self.wC*C + self.wR*R
    def threshold(self) -> float:
        if self.fixed_tau is not None:
            return float(self.fixed_tau)
        if self.count < self.min_events:
            return self.warmup_tau
        return float(self.mu + self.ksigma * math.sqrt(max(1e-6, self.var)))
    def update_stats(self, G: float):
        b = self.beta; self.mu = b*self.mu + (1-b)*G; self.var = b*self.var + (1-b)*(G - self.mu)**2; self.count += 1

# ---------------- prefix tags ----------------
class StatePrefixGenerator(nn.Module):
    def __init__(self, d_state: int, max_tokens: int = 32):
        super().__init__()
        self.max_tokens = max_tokens
        self.to_tags = nn.Sequential(nn.Linear(d_state, 128), nn.Tanh(), nn.Linear(128, 6))
        self.tag_vocab = ["calm","curious","precise","pragmatic","creative","cautious"]
    @torch.no_grad()
    def make_prefix(self, self_state: torch.Tensor) -> str:
        logits = self.to_tags(to_mod(self_state, self.to_tags))
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        top = np.argsort(-probs)[:3]
        tags = [self.tag_vocab[i] for i in top]
        return f"[STATE tags={','.join(tags)} intent=helpful memory=enabled]\n"

# ---------------- main sidecar (fast‑weights + affect) ----------------
class SidecarV3(nn.Module):
    def __init__(self, model, tokenizer, device="cpu", d_model=None, d_key=256, d_state=512, rank=64, key_topk=32, decay=0.995, eta0=0.1, temp=0.5,
                 affect_weights=(0.3,0.15,0.5,0.05,0.0), fixed_tau: Optional[float]=None, min_events: int = 2, warmup_tau: float = 0.90, s_mode: str = 'blend', s_alpha: float = 0.6):
        super().__init__()
        self.model = model.to(device); self.tok = tokenizer; self.device = torch.device(device)
        d_model = d_model or getattr(getattr(model, 'config', None), 'n_embd', 768)
        self.q_proj = nn.Sequential(nn.Linear(d_model, d_key), nn.Tanh()).to(self.device)
        self.self_state = SelfStateManager(d_state=d_state, d_in=d_key).to(self.device)
        self.prefixer = StatePrefixGenerator(d_state=d_state).to(self.device)
        self.fast = FastWeightsMemory(d=d_key, rank=rank, decay=decay, eta0=eta0, temp=temp, key_topk=key_topk).to(self.device)
        self.elig = EligBuffer(); self.affect = AffectScorer()
        wS,wN,wA,wC,wR = affect_weights
        self.gate = SurpriseGate(wS=wS,wN=wN,wA=wA,wC=wC,wR=wR,fixed_tau=fixed_tau,min_events=min_events,warmup_tau=warmup_tau)
        self.s_mode = s_mode; self.s_alpha = float(s_alpha)
        self.known_facts: Dict[str,str] = {}
        self.fact_color_re = re.compile(r"favorite\s+color\s+is\s+([A-Za-z]+)", re.I)
    @torch.no_grad()
    def update_known_facts_from_text(self, text: str):
        m = self.fact_color_re.search(text)
        if m:
            self.known_facts['favorite_color'] = m.group(1).lower()
    @torch.no_grad()
    def _encode_feat(self, text: str, max_len: int = 256):
        tokens = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        base = _get_base_module(self.model); out = base(**tokens); h = out.last_hidden_state
        k = self.q_proj(h[:, -1, :])
        return l2_normalize(k)
    @torch.no_grad()
    def _nll_surprise(self, text: str, max_len: int = 256) -> float:
        toks = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        labels = toks["input_ids"].clone(); out = self.model(**toks, labels=labels)
        loss = float(out.loss.detach().cpu().item()); return math.tanh(loss / 3.0)
    @torch.no_grad()
    def _surprise_novelty_read(self, k: torch.Tensor, text: str):
        r = self.fast.read(k); cos = float(cosine_sim(k, r).clamp(-1,1).item()); N = (1.0 - cos) * 0.5
        S_nll = self._nll_surprise(text)
        if self.s_mode == 'novelty': S = N
        elif self.s_mode == 'nll': S = S_nll
        else: S = self.s_alpha * S_nll + (1.0 - self.s_alpha) * N
        return S, N, r
    @torch.no_grad()
    def _conflict(self, text: str) -> float:
        m = self.fact_color_re.search(text)
        if m:
            new = m.group(1).lower(); old = self.known_facts.get('favorite_color')
            if old and old != new:
                return 1.0
        return 0.0
    @torch.no_grad()
    def _write(self, text: str) -> Dict[str, float]:
        k = self._encode_feat(text)
        S, N, r = self._surprise_novelty_read(k, text)
        A = float(self.affect.arousal(text))
        C = self._conflict(text)
        v_resid = l2_normalize((k - r))
        G = self.gate.score(S, N, A, C, 0.0)
        self.gate.update_stats(G)
        committed = 0.0
        if G > self.gate.threshold():
            self.fast.commit(k, v_resid, gain=G)
            for e in self.elig.pop_alive():
                self.fast.commit(e.k.to(self.device), e.v.to(self.device), gain=G)
            committed = 1.0
        else:
            self.elig.add(k, v_resid, S)
        # update internal facts after deciding commit (so conflict reflects prior state)
        self.update_known_facts_from_text(text)
        return {"S": S, "N": N, "A": A, "C": C, "G": G, "tau": self.gate.threshold(), "committed": committed}
    @torch.no_grad()
    def _assemble_prompt(self, user_prompt: str):
        k = self._encode_feat(user_prompt); r = self.fast.read(k); self.self_state(r)
        return self.prefixer.make_prefix(self.self_state.get()) + user_prompt
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, top_p: float = 0.9):
        assembled = self._assemble_prompt(prompt); _ = self._write(assembled)
        inputs = self.tok(assembled, return_tensors="pt").to(self.device)
        do_sample = (temperature is not None) and (float(temperature) > 0.0)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.eos_token_id, "eos_token_id": self.tok.eos_token_id, "use_cache": True, "do_sample": do_sample}
        if do_sample: gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        out = self.model.generate(**inputs, **gen_kwargs); gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen_ids, skip_special_tokens=True)
    @torch.no_grad()
    def _token_ids_for_words(self, words: List[str]) -> List[int]:
        ids = set()
        for w in words:
            for v in [w, w.capitalize(), f" {w}", f" {w.capitalize()}"]:
                tok = self.tok.encode(v, add_special_tokens=False)
                if len(tok) == 1: ids.add(tok[0])
        return list(ids)
    @torch.no_grad()
    def generate_kv(self, mem_text: str, prompt: str, max_new_tokens: int = 16, stop_on_newline: bool = False, allowed_words: Optional[List[str]] = None):
        device = self.device
        mem_ids = self.tok(mem_text, return_tensors="pt").input_ids.to(device)
        out = self.model(input_ids=mem_ids, use_cache=True); past = out.past_key_values
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(device)
        out = self.model(input_ids=p_ids, past_key_values=past, use_cache=True); past = out.past_key_values; logits = out.logits[:, -1, :]
        eos_id = self.tok.eos_token_id; nl_ids = self.tok.encode("\n", add_special_tokens=False) if stop_on_newline else []
        allow = None
        if allowed_words: allow = set(self._token_ids_for_words(allowed_words))
        gen = []
        for _ in range(max_new_tokens):
            next_logits = logits.clone()
            if allow:
                mask = torch.full_like(next_logits, -1e9); idxs = list(allow)
                mask[:, idxs] = next_logits[:, idxs]; next_logits = mask
            nid = torch.argmax(next_logits, dim=-1)
            if int(nid) == eos_id or (nl_ids and int(nid) in nl_ids): break
            gen.append(int(nid))
            out = self.model(input_ids=nid.view(1,1), past_key_values=past, use_cache=True)
            past = out.past_key_values; logits = out.logits[:, -1, :]
        return self.tok.decode(gen, skip_special_tokens=True)
    def save_state(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.self_state.state.detach().cpu(), os.path.join(path, "self_state.pt"))
        torch.save({"U": self.fast.U.detach().cpu(), "V": self.fast.V.detach().cpu()}, os.path.join(path, "fastweights.pt"))
        with open(os.path.join(path, "gate.json"), "w") as f:
            json.dump({"mu": getattr(self.gate,'mu',0.0), "var": getattr(self.gate,'var',0.0), "count": getattr(self.gate,'count',0)}, f)
    def load_state(self, path: str):
        self.self_state.state.data = torch.load(os.path.join(path, "self_state.pt"), map_location=self.device)
        fw = torch.load(os.path.join(path, "fastweights.pt"), map_location=self.device)
        self.fast.U.data.copy_(fw["U"].to(self.device)); self.fast.V.data.copy_(fw["V"].to(self.device))
        try:
            with open(os.path.join(path, "gate.json"), "r") as f:
                g = json.load(f); self.gate.mu = float(g.get("mu", self.gate.mu)); self.gate.var = float(g.get("var", self.gate.var)); self.gate.count = int(g.get("count", self.gate.count))
        except Exception: pass
''')

with open('sidecar_memory.py', 'w') as f:
    f.write(sidecar_code)
print('✅ Wrote sidecar_memory.py (v0.3.3)')

# ==============================
# 2) Demo: affect‑gated surprise vs ordinary + color recall
# ==============================
import re, json, os, sys, importlib

class SimpleFactStore:
    def __init__(self, path='./demo_mem_state/facts.json'):
        self.path = path; self.data = {}; self.load()
    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r') as f: self.data = json.load(f)
        except Exception: self.data = {}
    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f: json.dump(self.data, f)
    def update_from_text(self, text: str):
        m = re.search(r"favorite\s+color\s+is\s+([A-Za-z]+)", text, flags=re.I)
        if m: self.data['favorite_color'] = m.group(1).lower()
    def render_block(self):
        if not self.data: return ''
        return f"The user's favorite color is {self.data.get('favorite_color','')}\.\n"

# Force‑reload
if 'sidecar_memory' in sys.modules:
    importlib.reload(sys.modules['sidecar_memory'])
import sidecar_memory as scm
print('sidecar_memory version:', getattr(scm, '__version__', 'unknown'))
SidecarV3 = scm.SidecarV3

# ==============================
# 3) Run the demo
# ==============================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'gpt2'
print(f'Loading {MODEL_NAME} on {DEVICE}...')

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.eos_token is None:
    tok.eos_token = tok.pad_token = tok.sep_token = '<|endoftext|>'
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

mem = SidecarV3(
    model, tok, device=DEVICE,
    d_key=256, rank=64, key_topk=32, decay=0.997, eta0=0.15, temp=0.5,
    affect_weights=(0.3,0.15,0.5,0.05,0.0), fixed_tau=None, min_events=2, warmup_tau=0.90,
    s_mode='blend', s_alpha=0.6
)
facts = SimpleFactStore()

KNOWN_COLORS = { 'blue','red','green','yellow','orange','purple','pink','black','white','gray','grey','brown','cyan','magenta','violet','indigo','teal','maroon','beige','silver','gold','navy','lime' }

def extract_one_word(s: str) -> str:
    for w in re.findall(r"[A-Za-z]+", s):
        lw = w.lower()
        if lw in KNOWN_COLORS: return lw
    m = re.search(r"\b([A-Za-z]{3,20})\b", s)
    return (m.group(1).lower() if m else s.strip()[:20])

print('\n=== Session 1 — Affect‑gated surprise ===')
boring = 'I ate a sandwich for lunch today. It tasted fine.'
surprising = 'I drove through Death Valley in 117 degree heat and the car thermometer flashed a warning.'

print('\nUSER:', boring)
stats1 = mem._write(boring)
print('Write stats:', stats1)
print('ASSISTANT:', mem.generate(boring, max_new_tokens=24))

print('\nUSER:', surprising)
stats2 = mem._write(surprising)
print('Write stats:', stats2)
print('ASSISTANT:', mem.generate(surprising, max_new_tokens=24))

print('\n=== Fact recall with fast‑weights in the loop ===')
teach = 'Remember that my favorite color is blue.'
print('USER:', teach)
facts.update_from_text(teach); facts.save(); mem.update_known_facts_from_text(teach)
print('ASSISTANT:', mem.generate(teach + "\n" + facts.render_block(), max_new_tokens=24))

ask = 'What color did I say I liked? Answer with just the color.'
print('\nUSER:', ask)
kv_ans = mem.generate_kv(facts.render_block(), ask, max_new_tokens=4, stop_on_newline=False, allowed_words=sorted(list(KNOWN_COLORS)))
print('ASSISTANT (KV):', extract_one_word(kv_ans))

SAVE_DIR = './demo_mem_state_v3'
mem.save_state(SAVE_DIR); facts.save(); print(f'💾 Saved state to {SAVE_DIR}')

print('\n=== New session (state reloaded) ===')
new_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
new_mem = SidecarV3(new_model, tok, device=DEVICE)
new_mem.load_state(SAVE_DIR)
print('USER:', ask)
kv_ans2 = new_mem.generate_kv(facts.render_block(), ask, max_new_tokens=4, stop_on_newline=False, allowed_words=sorted(list(KNOWN_COLORS)))
print('ASSISTANT (KV):', extract_one_word(kv_ans2))
