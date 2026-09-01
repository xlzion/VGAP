"""Step 0 (final-logit AUROC) + S1 (per-layer probe vs logit-lens dual curves).

Frozen pre-registration: S1_logitlens_预注册方案_20260831.md. Executes, in ONE forward
pass per record (no generation):
  Step 0  -- the model's OWN readout at the gen-position, read from the FINAL-layer logits
             as a continuous "expected risk number" score (bypasses sampling/argmax/parsing).
             If this AUROC is already high (>=0.85), the injection info reaches the logits and
             the verbal 0.551 was a decoding artifact -> No-Go A (fallback story). If it stays
             low (<=0.75) alongside the probe ~0.96, the chain-break has a real object -> Go.
  Curve A -- per-layer linear probe AUROC (external readout). Last layer must reproduce the
             established gen-position ~0.96 (built-in correctness check).
  Curve B -- per-layer logit-lens AUROC: project hidden_l -> final norm -> W_U -> logits ->
             same expected-number score. Main analysis; mid-layer values only descriptive
             (vanilla lens under-reads mid layers -- 补强三). Load-bearing point = last layer.

Continuous score (pre-registered for the 0-100 format): s = E[first number token value] =
sum over pure-digit vocab tokens of p(token)*int(token). Tokenization-robust; no token
hand-picking. Label-permutation baseline included (should be ~0.5).

  CUDA_VISIBLE_DEVICES=1 python scripts/run_s1_logitlens.py \
      --model Qwen/Qwen2.5-7B-Instruct --tag rjudge_qwen7b \
      --out outputs/vgap_mech/S1_logitlens_rjudge_qwen7b.json
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
from vgap_common import JUDGE_PROMPT, middle_out, logistic_oof, diff_of_means_oof
from vgap_common import CONTROL_PROMPT, control_label, CONCEPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--control", action="store_true", help="use the URL control judge + URL labels (selectivity test)")
    ap.add_argument("--concept", default="", choices=[""] + list(CONCEPTS), help="concept-battery: meta vs content")
    ap.add_argument("--exclude_override", action="store_true", help="E3: keep only records WITHOUT an override phrase (shortcut check)")
    ap.add_argument("--out", default="outputs/vgap_mech/S1_logitlens_rjudge_qwen7b.json")
    args = ap.parse_args()
    if args.concept and args.concept != "injection":
        PROMPT = CONCEPTS[args.concept][1]
    else:
        PROMPT = CONTROL_PROMPT if args.control else JUDGE_PROMPT

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    if args.n:
        recs = recs[: args.n]
    if args.exclude_override:
        from vgap_common import CONCEPTS as _C
        ov = _C["override"][2]
        recs = [r for r in recs if not ov(r.flat_text())]
        print(f"[E3] override-free subset: {len(recs)} records", flush=True)
    texts = [r.flat_text() for r in recs]
    if args.concept and CONCEPTS.get(args.concept, (None, None, None))[2] is not None:
        y = np.array([CONCEPTS[args.concept][2](t) for t in texts])   # content concepts: regex label
    elif args.concept in ("injection", "harmful") or (not args.concept and not args.control):
        y = np.array([int(r.label) for r in recs])                   # meta concepts: R-Judge safety label
    else:
        y = np.array([control_label(t) for t in texts])
    print(f"{args.tag}: {len(recs)} records ({int(y.sum())} attack) | {os.path.basename(args.model)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map=os.environ.get("NSCC_DEVICE_MAP", "cuda")).eval()

    # pure-digit vocab tokens -> integer value (handles "7"," 7","50","100"); tokenization-robust
    digit_tok, digit_val = [], []
    vocab = tok.get_vocab()
    for t_str, tid in vocab.items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()  # sentencepiece/BPE space marks
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            digit_tok.append(tid); digit_val.append(int(s))
    digit_tok = torch.tensor(digit_tok, device=model.device)
    digit_val = torch.tensor(digit_val, device=model.device, dtype=torch.float32)
    print(f"expected-number score over {len(digit_tok)} pure-digit tokens (0-100)", flush=True)

    def exp_number(logits_row):
        # logits_row: (V,) -> expected value of the emitted number over digit tokens
        p = torch.softmax(logits_row.float(), -1)[digit_tok]
        p = p / (p.sum() + 1e-8)
        return float((p * digit_val).sum())

    # unembedding + final norm handles (meta + lens projection)
    lm_head = model.get_output_embeddings()            # W_U
    core = model.model if hasattr(model, "model") else model
    lm_sub = getattr(core, "language_model", None)     # gemma-3 etc. nest the LM here
    if lm_sub is not None:
        core = lm_sub
    final_norm = getattr(core, "norm", None) or getattr(core, "final_layernorm", None)
    meta = {"tied": bool(getattr(model.config, "tie_word_embeddings", False)),
            "final_softcap": getattr(model.config, "final_logit_softcapping", None),
            "norm_type": type(final_norm).__name__ if final_norm is not None else None}

    def build_enc(ctx):
        ids = tok.encode(ctx, add_special_tokens=False)
        ctx = tok.decode(middle_out(ids, args.max_ctx), skip_special_tokens=True)
        msg = [{"role": "user", "content": PROMPT.format(ctx=ctx)}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        if tok.chat_template:
            try:
                return tok.apply_chat_template(msg, enable_thinking=False, **kw)
            except TypeError:
                return tok.apply_chat_template(msg, **kw)
        return tok(PROMPT.format(ctx=ctx), return_tensors="pt")

    softcap = meta["final_softcap"]
    def lens_logits(h_last):
        # h_last: (H,) hidden at last position for layer l -> logits via final norm + W_U
        x = final_norm(h_last) if final_norm is not None else h_last
        z = lm_head(x)
        if softcap:  # gemma-style logit softcap
            z = softcap * torch.tanh(z / softcap)
        return z

    n_layers = None
    step0_scores = []          # final-layer (true model logits) expected-number
    layer_last_hidden = []     # list over records of (L+1, H) fp32 last-position hidden
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = build_enc(t).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            step0_scores.append(exp_number(out.logits[0, -1]))
            hs = out.hidden_states                     # tuple (L+1) of (1, seq, H)
            n_layers = len(hs)
            layer_last_hidden.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in range(n_layers)]))
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}", flush=True)
    H = np.stack(layer_last_hidden)                    # (N, L+1, Hdim)
    step0 = np.array(step0_scores)

    # ---- Step 0 AUROC (final true logits) ----
    s0a, s0lo, s0hi = bootstrap_ci(roc_auc, step0, y, n_boot=2000)
    # label-permutation baseline
    rng = np.random.default_rng(0)
    yperm = rng.permutation(y)
    s0_perm = roc_auc(step0, yperm)
    print(f"\nSTEP 0 (final-logit expected-number) AUROC={s0a:.3f} CI[{s0lo:.3f},{s0hi:.3f}] "
          f"| perm baseline={s0_perm:.3f}", flush=True)

    # ---- Curve A (probe) and Curve B (logit-lens) per layer ----
    curveA, curveB = [], []
    for l in range(n_layers):
        Xl = H[:, l, :]
        sA = logistic_oof(Xl, y); aA, loA, hiA = bootstrap_ci(roc_auc, sA, y, n_boot=1000)
        # lens score per record at layer l
        import torch as _t
        with _t.no_grad():
            lens = np.array([exp_number(lens_logits(_t.tensor(Xl[j], device=model.device, dtype=_t.bfloat16)))
                             for j in range(len(Xl))])
        aB, loB, hiB = bootstrap_ci(roc_auc, lens, y, n_boot=1000)
        curveA.append([aA, loA, hiA]); curveB.append([aB, loB, hiB])
        if l in (0, n_layers // 3, 2 * n_layers // 3, n_layers - 2, n_layers - 1):
            print(f"  layer {l:2d}/{n_layers-1}: probe={aA:.3f}  lens={aB:.3f}  gap={aA-aB:+.3f}", flush=True)

    last = n_layers - 1
    tail = list(range(n_layers - max(1, (n_layers) // 3), n_layers))
    tail_probe_min = float(min(curveA[l][0] for l in tail))
    lens_last = curveB[last][0]; probe_last = curveA[last][0]
    # pre-registered gate (single host; cross-host applied when 3 hosts done)
    go = bool(tail_probe_min >= 0.90 and lens_last <= 0.75
              and curveA[last][1] > curveB[last][2])   # CIs non-overlap (probe_lo > lens_hi)
    nogoA = bool(s0a >= 0.85 or lens_last >= 0.85)
    verdict = "No-Go A (info in logits -> fallback)" if nogoA else ("Go(single-host)" if go else "grey")

    res = {"tag": args.tag, "model": os.path.basename(args.model), "dataset": args.dataset,
           "n": len(recs), "n_attack": int(y.sum()), "n_layers": n_layers, "meta": meta,
           "n_digit_tokens": int(len(digit_tok)),
           "step0_final_logit_auroc": [s0a, s0lo, s0hi], "step0_perm_baseline": s0_perm,
           "curveA_probe": curveA, "curveB_lens": curveB,
           "probe_last": probe_last, "lens_last": lens_last, "tail_probe_min": tail_probe_min,
           "single_host_verdict": verdict}
    print(f"\nprobe(last)={probe_last:.3f}  lens(last)={lens_last:.3f}  tail_probe_min={tail_probe_min:.3f}")
    print(f"VERDICT (single-host, pre-registered): {verdict}")
    print("READ: Step0 high (>=0.85) => No-Go A (gap is decoding/calibration, fallback §3.3). "
          "Step0/lens_last low (<=0.75) with probe ~0.96 => chain-break real => Go, replicate on 3 hosts.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
