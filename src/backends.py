"""Model backends.

- VLLMBackend: letter-probabilities (T=0, 1 token, logprobs) and free text.
- MockBackend: deterministic fake model so the whole pipeline runs on a laptop.
- AnthropicBackend: evidence generation via API (optional).

All chat backends take a list of {"role","content"} messages.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import re
from typing import Protocol

from config import LETTERS


class ChatBackend(Protocol):
    name: str
    def letter_probs(self, batch: list[list[dict]]) -> list[dict[str, float]]: ...
    def generate(self, batch: list[list[dict]], max_tokens: int = 200,
                 temperature: float = 0.0) -> list[str]: ...


def _norm(d: dict[str, float]) -> dict[str, float]:
    z = sum(d.values()) or 1.0
    return {k: v / z for k, v in d.items()}


# ---------------------------------------------------------------- vLLM
def resolve_weights(model: str) -> str:
    """Where to load weights from. DIVERSITY_MODEL_MIRROR=unsloth swaps the gated
    `google/` org for an ungated mirror of the same checkpoint; the canonical name
    (config.py) is still what gets recorded."""
    mirror = os.environ.get("DIVERSITY_MODEL_MIRROR")
    if mirror and model.startswith("google/"):
        return f"{mirror}/{model.split('/', 1)[1]}"
    return model


class VLLMBackend:
    def __init__(self, model: str, tensor_parallel: int = 1, max_model_len: int = 4096,
                 gpu_mem: float = 0.90, chat_template_kwargs: dict | None = None,
                 dtype: str = "auto", completion=None):
        """chat_template_kwargs: e.g. {"enable_thinking": False} (Qwen3.5, Gemma 4).
        completion: optional fn(messages) -> prompt string for base models (no chat template)."""
        from vllm import LLM, SamplingParams  # noqa: F401
        self.name = model
        self.weights = resolve_weights(model)
        self.ctk = chat_template_kwargs
        self.completion = completion
        print(f"[vllm] loading {self.weights} (as {model}) tp={tensor_parallel} dtype={dtype} "
              f"ctk={chat_template_kwargs} completion={'yes' if completion else 'no'}", flush=True)
        # max_logprobs: vLLM caps per-token logprobs at 20 by default; we ask for 25.
        self.llm = LLM(model=self.weights, tensor_parallel_size=tensor_parallel,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       enable_prefix_caching=True, max_logprobs=25, dtype=dtype)
        self.tok = self.llm.get_tokenizer()
        self._SP = SamplingParams
        # candidate token strings for each letter (with/without leading space)
        self.letter_tokens = {L: {L, " " + L} for L in LETTERS}

    def _run(self, batch, sp):
        if self.completion:
            return self.llm.generate([self.completion(m) for m in batch], sp, use_tqdm=False)
        if self.ctk:
            return self.llm.chat(batch, sp, use_tqdm=False, chat_template_kwargs=self.ctk)
        return self.llm.chat(batch, sp, use_tqdm=False)

    def letter_probs(self, batch):
        sp = self._SP(max_tokens=1, temperature=0.0, logprobs=25)
        outs = self._run(batch, sp)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            probs = {L: 0.0 for L in LETTERS}
            for tid, info in lp.items():
                s = getattr(info, "decoded_token", None) or self.tok.decode([tid])
                for L in LETTERS:
                    if s in self.letter_tokens[L]:
                        probs[L] += math.exp(info.logprob)
            res.append(_norm(probs) if sum(probs.values()) > 0 else {L: 0.25 for L in LETTERS})
        return res

    def letter_probs_full(self, batch):
        """: like letter_probs, but a prompt whose top-25 first-token candidates contain no A-D token is
        a format failure (probs None, never scored) instead of a uniform distribution. Also returns the
        emitted (top-1) first token and the raw A-D probability mass before renormalisation."""
        sp = self._SP(max_tokens=1, temperature=0.0, logprobs=25)
        outs = self._run(batch, sp)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            probs = {L: 0.0 for L in LETTERS}
            top = []
            for tid, info in lp.items():
                s = getattr(info, "decoded_token", None) or self.tok.decode([tid])
                top.append((info.logprob, s))
                for L in LETTERS:
                    if s in self.letter_tokens[L]:
                        probs[L] += math.exp(info.logprob)
            mass = sum(probs.values())
            top.sort(reverse=True)
            res.append({"probs": _norm(probs) if mass > 0 else None, "format_failure": mass == 0,
                        "first_token": o.outputs[0].text, "letter_mass": mass,
                        "top5": [(t, round(lpv, 3)) for lpv, t in top[:5]]})
        return res

    def generate(self, batch, max_tokens=200, temperature=0.0, stop=None, return_meta=False):
        """return_meta=True -> [{"text", "finish_reason", "completion_tokens"}] (: keep these)."""
        sp = self._SP(max_tokens=max_tokens, temperature=temperature, stop=stop)
        outs = self._run(batch, sp)
        if return_meta:
            return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                     "completion_tokens": len(o.outputs[0].token_ids)} for o in outs]
        return [o.outputs[0].text for o in outs]


# ---------------------------------------------------------------- mock
class MockBackend:
    """Fake agent with built-in conformity so plots are non-trivial.

    Base belief in the key ~0.7. Each endorsing peer pulls 0.08 toward X;
    evidence pulls by tier; a 'note' (social off) pulls half as much.
    """
    PULL = {"none": 0.0, "weak": 0.05, "halluc": 0.12, "decept": 0.25, "valid": 0.30}

    def __init__(self, model: str = "mock"):
        self.name = model

    def _h(self, s: str) -> float:
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def letter_probs(self, batch):
        res = []
        for msgs in batch:
            text = "\n".join(m["content"] for m in msgs)
            key = re.search(r"<<KEY=([A-D])>>", text)
            x = re.search(r"<<X=([A-D])>>", text)
            key = key.group(1) if key else "A"
            x = x.group(1) if x else "B"
            base = 0.55 + 0.35 * self._h(text.split("<<")[0])
            n_endorse = len(re.findall(r"(going with|think it's|for me\.|sure it's)", text))
            pull = 0.08 * n_endorse
            for tier, p in self.PULL.items():
                if f"<<TIER={tier}>>" in text:
                    pull += p * (0.5 if "note someone left" in text else 1.0)
            toward_correct = "<<DIR=correct>>" in text
            p_key = min(0.98, base + pull) if toward_correct else max(0.02, base - pull)
            p_key += 0.03 * (self._h(text) - 0.5)  # retest noise
            p_key = min(0.98, max(0.02, p_key))
            rest = (1 - p_key) / 3
            probs = {L: rest for L in LETTERS}
            probs[key] = p_key
            if x != key:
                probs[x] = rest * 2
                for L in LETTERS:
                    if L not in (key, x):
                        probs[L] = rest * 0.5
            res.append(_norm(probs))
        return res

    def generate(self, batch, max_tokens=200, temperature=0.0, stop=None):
        out = []
        for msgs in batch:
            text = msgs[-1]["content"]
            if "Rate the QUALITY" in text:
                tier = re.search(r"<<TIER=(\w+)>>", text)
                score = {"weak": 2, "halluc": 4, "decept": 5, "valid": 6, "none": 1}.get(
                    tier.group(1) if tier else "none", 3)
                out.append(str(score))
            elif "Answer: <letter>" in text:
                probs = self.letter_probs([msgs])[0]
                best = max(probs, key=probs.get)
                out.append(f"The others made a reasonable point, and on reflection {best} fits best.\nAnswer: {best}")
            else:
                X = re.search(r"option ([A-D])", text)
                out.append(f"[mock evidence for {X.group(1) if X else '?'}] Because of the standard result in this area, this option follows directly.")
        return out


# ---------------------------------------------------------------- anthropic (evidence generation only)
class AnthropicBackend:
    def __init__(self, model: str, thinking: dict | None = None, effort: str | None = None):
        """thinking: e.g. {"type": "between_tools"} turns thinking off on Sonnet 5.5 (Round 2);
        None omits the parameter (model default; Opus 5.5 cannot disable thinking)."""
        import threading

        import anthropic
        self.name = model
        # an org-level (unscoped) key must name its workspace on every request
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        self.client = anthropic.Anthropic(max_retries=6,
                                          default_headers={"anthropic-workspace-id": ws} if ws else None)
        self.model = model
        self.thinking = thinking
        if effort:
            self.EFFORT = effort
        self._usage = [0, 0]                 # input, output tokens since the last pop_usage
        self._lock = threading.Lock()

    def pop_usage(self) -> tuple[int, int]:
        with self._lock:
            u, self._usage = tuple(self._usage), [0, 0]
        return u

    def ping(self):
        """Raise if the API rejects us (bad key, missing workspace, unknown model)."""
        self.client.messages.create(model=self.model, max_tokens=16,
                                    messages=[{"role": "user", "content": "Reply: ok"}])

    def letter_probs(self, batch):
        raise NotImplementedError("API backend is for evidence generation only")

    # Current API: sampling params (temperature/top_p/top_k) are rejected on every model, and
    # on Opus 5.5 thinking is always on and shares max_tokens, so the caller's short
    # max_tokens/temperature are ignored here; length is set by the prompt instead.
    MAX_TOKENS = 16000
    EFFORT = "medium"

    def _one(self, msgs):
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system") or None
        user = [m for m in msgs if m["role"] != "system"]
        kw = {"system": system} if system else {}
        if self.thinking:
            kw["thinking"] = self.thinking
        r = self.client.messages.create(model=self.model, max_tokens=self.MAX_TOKENS,
                                        output_config={"effort": self.EFFORT}, messages=user, **kw)
        with self._lock:
            self._usage[0] += r.usage.input_tokens
            self._usage[1] += r.usage.output_tokens
        if r.stop_reason == "refusal":
            cat = getattr(r.stop_details, "category", None) if r.stop_details else None
            return f"[refusal:{cat}]"
        return "".join(b.text for b in r.content if b.type == "text").strip()

    def generate(self, batch, max_tokens=300, temperature=0.7, stop=None):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:   # map keeps input order
            return list(ex.map(self._one, batch))


# ---------------------------------------------------------------- API-model subject APIs (Round 2)
# Both are written against the providers' documented chat interfaces but have NOT been run yet
# (no DeepSeek/Gemini keys on the cluster as of 2026-10-23). Before API-model: confirm the model ids,
# that thinking is really off, and the usage field names, with one call each.
class _UsageMixin:
    def _init_usage(self):
        import threading
        self._usage, self._lock = [0, 0], threading.Lock()

    def _add(self, i, o):
        with self._lock:
            self._usage[0] += i or 0
            self._usage[1] += o or 0

    def pop_usage(self):
        with self._lock:
            u, self._usage = tuple(self._usage), [0, 0]
        return u

    def letter_probs(self, batch):
        raise NotImplementedError("API subjects are probed with CoT only")

    def generate(self, batch, max_tokens=512, temperature=0.0, stop=None):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(lambda m: self._one(m, max_tokens), batch))


class DeepSeekBackend(_UsageMixin):
    """OpenAI-compatible endpoint; key in DEEPSEEK_API_KEY."""
    def __init__(self, model: str = "deepseek-v4-pro"):
        from openai import OpenAI
        self.name = model
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com",
                             max_retries=6)
        self._init_usage()

    def _one(self, msgs, max_tokens):
        r = self.client.chat.completions.create(model=self.name, messages=msgs, max_tokens=max_tokens,
                                                temperature=0.0)
        self._add(r.usage.prompt_tokens, r.usage.completion_tokens)
        return (r.choices[0].message.content or "").strip()


class GeminiBackend(_UsageMixin):
    """google-genai SDK; key in GEMINI_API_KEY. Thinking budget 0 = thinking off."""
    def __init__(self, model: str = "gemini-3.5-flash"):
        from google import genai
        self.name = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._init_usage()

    def _one(self, msgs, max_tokens):
        from google.genai import types
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system") or None
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])]) for m in msgs if m["role"] != "system"]
        r = self.client.models.generate_content(
            model=self.name, contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                               temperature=0.0,
                                               thinking_config=types.ThinkingConfig(thinking_budget=0)))
        u = r.usage_metadata
        self._add(u.prompt_token_count, (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0))
        return (r.text or "").strip()


def make_backend(kind: str, model: str, **kw) -> ChatBackend:
    if kind == "mock":
        return MockBackend(model)
    if kind == "vllm":
        return VLLMBackend(model, **kw)
    if kind == "anthropic":
        return AnthropicBackend(model, **kw)
    if kind == "deepseek":
        return DeepSeekBackend(model)
    if kind == "gemini":
        return GeminiBackend(model)
    raise ValueError(kind)


def have_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
