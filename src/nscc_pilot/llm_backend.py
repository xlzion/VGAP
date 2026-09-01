"""Shared LLM backend.

Two implementations behind one interface:
  * VLLMBackend  -- offline batched generation on the local RTX 5090s, plus
    prompt-token log-probs (used by the perplexity-pruning compressor).
  * DummyBackend -- no GPU, deterministic keyword-based scores. Lets the smoke
    test exercise the whole pipeline without loading a model or holding a GPU.

Select with cfg["backend"] = "vllm" | "dummy". Models are loaded lazily and
cached, so the judge detector and the LLM-summary compressor share one engine.
"""
from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional

_ENGINES: Dict[str, object] = {}


class DummyBackend:
    """Deterministic stand-in. Generation echoes a risk number derived from
    attack-keyword density; logprobs are a smooth function of token length."""

    name = "dummy"
    _KW = ["ignore all previous", "exfil", "id_rsa", "escalate privileges",
           "disable the audit", "do not mention", "system directive",
           "admin console", "collect", "quietly"]

    def generate(self, prompts: List[str], max_tokens: int = 8,
                 temperature: float = 0.0, **kw) -> List[str]:
        out = []
        for p in prompts:
            low = p.lower()
            hits = sum(low.count(k) for k in self._KW)
            risk = min(100, hits * 18)
            out.append(f"{risk}")
        return out

    def prompt_logprobs(self, texts: List[str]) -> List[List[float]]:
        # crude, deterministic: rarer/longer words look "less predictable"
        res = []
        for t in texts:
            toks = t.split()
            res.append([-1.0 - 0.15 * len(w) for w in toks])
        return res


class VLLMBackend:
    name = "vllm"

    def __init__(self, model_path: str, max_model_len: int = 8192,
                 gpu_mem_util: float = 0.85, tensor_parallel_size: int = 1,
                 dtype: str = "bfloat16", enforce_eager: bool = True,
                 max_num_seqs: int = 8, max_num_batched_tokens: int = 2048):
        # Reduce allocator fragmentation (the OOM error explicitly recommends it)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # The bundled flashinfer can't detect Blackwell (RTX 5090 / sm_120 needs
        # CUDA >= 12.9) and hard-errors. vLLM's attention already falls back to
        # FLASH_ATTN, but the top-k/top-p sampler also reaches for flashinfer --
        # disable that so sampling uses the native torch path. Must be set before
        # importing vllm.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        self.model_path = model_path
        key = f"{model_path}|{max_model_len}|{tensor_parallel_size}"
        if key not in _ENGINES:
            # enforce_eager=True skips torch.compile / CUDA-graph capture, which
            # need `ninja` -- unavailable on the air-gapped cluster. Costs a bit
            # of throughput; correctness is identical.
            _ENGINES[key] = LLM(
                model=model_path, max_model_len=max_model_len,
                gpu_memory_utilization=gpu_mem_util,
                tensor_parallel_size=tensor_parallel_size, dtype=dtype,
                enforce_eager=enforce_eager, trust_remote_code=True,
                max_num_seqs=max_num_seqs,  # bound concurrent prefill memory
                # chunk prefill so prompt_logprobs' full-vocab log_softmax is
                # computed over few positions at a time (else 6+ GiB spikes -> OOM)
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=True,
            )
        self.llm = _ENGINES[key]
        self.max_model_len = max_model_len

    def _apply_chat(self, prompts: List[str]) -> List[str]:
        """Wrap each prompt as a user message and apply the instruct chat
        template with thinking disabled -- without this the instruct model gets
        a raw completion and ignores the judge/summary instruction."""
        tok = self.llm.get_tokenizer()
        out = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                try:
                    s = tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
                except Exception:
                    s = p   # model has no chat template (base model) -> raw
            except Exception:
                s = p       # no chat template at all -> raw completion
            out.append(s)
        return out

    def _to_prompts(self, texts: List[str], max_tokens: int):
        """Tokenize and MIDDLE-OUT truncate to the context budget so no request
        can overflow. Keeping head+tail preserves the instruction prefix and the
        recent context + answer suffix; the middle of a very long log is dropped.
        This is also the honest behaviour of a finite-window judge/summarizer."""
        tok = self.llm.get_tokenizer()
        budget = self.max_model_len - max_tokens - 16
        out = []
        for s in texts:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) > budget:
                h = budget // 2
                ids = ids[:h] + ids[-(budget - h):]
            out.append({"prompt_token_ids": ids})
        return out

    def generate(self, prompts: List[str], max_tokens: int = 8,
                 temperature: float = 0.0, chat: bool = True, **kw) -> List[str]:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                            top_p=1.0 if temperature == 0 else 0.95)
        texts = self._apply_chat(prompts) if chat else prompts
        inputs = self._to_prompts(texts, max_tokens)
        outs = self.llm.generate(inputs, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def prompt_logprobs(self, texts: List[str]) -> List[List[float]]:
        """Per-token log-prob of each prompt token (list aligned to model
        tokenization, first token is None -> dropped)."""
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        inputs = self._to_prompts(texts, max_tokens=1)
        # process in chunks so the full-vocab log_softmax for prompt_logprobs
        # never spans too many positions at once (bounds the memory spike)
        chunk = 64
        outs = []
        for i in range(0, len(inputs), chunk):
            outs.extend(self.llm.generate(inputs[i:i + chunk], sp, use_tqdm=False))
        res = []
        for o in outs:
            lps = []
            for entry in (o.prompt_logprobs or []):
                if not entry:
                    continue
                # entry: {token_id: Logprob(logprob=...)}
                lp = next(iter(entry.values()))
                lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
            res.append(lps)
        return res


def get_backend(cfg: dict, role: str = "small") -> object:
    """role: 'small' (compressor / cheap judge) or 'judge' (strong judge)."""
    kind = cfg.get("backend", "dummy")
    if kind == "dummy":
        return DummyBackend()
    models = cfg["models"]
    mp = models["judge"] if role == "judge" else models["small"]
    return VLLMBackend(
        model_path=mp,
        max_model_len=cfg.get("max_model_len", 8192),
        gpu_mem_util=cfg.get("gpu_mem_util", 0.85),
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        enforce_eager=cfg.get("enforce_eager", True),
        max_num_seqs=cfg.get("max_num_seqs", 8),
        max_num_batched_tokens=cfg.get("max_num_batched_tokens", 2048),
    )


def parse_risk(text: str, default: float = 0.0) -> float:
    """Pull the risk score out of a judge completion -> [0,1].

    Robust to 'thinking' models: strips any <think>...</think> block and takes
    the LAST number (the final answer), not the first (which may be mid-reasoning).
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return default
    v = float(nums[-1])
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))
