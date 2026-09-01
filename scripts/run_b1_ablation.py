"""B1: causal verification -- do the B0 erasers CAUSALLY silence the verbal report?

Intervention (the ONLY one, M-C discipline): mean-ablation of the top eraser (branch,layer)
components identified by B0, at the gen (last-prompt) position -- replace their branch output
with the BENIGN-class mean. This REMOVES the eraser's contribution (not steering, not a probe
direction; target comes from attribution, operation is removal toward benign mean).

Measures, per condition, vs 3 required controls:
  conditions: none (baseline) | eraser (top-K erasers) | random (K random branch,layer, matched)
              | noneraser (top-K most-POSITIVE dsep = writers)
  metrics:   (a) Step0 readout AUROC (expected-number from ablated final logits, no generation)
             (b) verbal AUROC -- the ACTUAL generated 0-100 judgment (main metric)
  side-effects (eraser condition): URL-control verbal AUROC (should not drop), format/parse rate.

Gate (pre-registered): verbal AUROC recovery (eraser - none) >= 0.15 AND exceeds both controls
(CI-separated), >=3 hosts same-direction -> headline. Partial: only readout recovers -> "erasure
at component, second gate on verbal". Fail / cross-host inconsistent -> appendix single-host.

  CUDA_VISIBLE_DEVICES=1 python scripts/run_b1_ablation.py --model Qwen/Qwen2.5-7B-Instruct \
      --b0 outputs/vgap_mech/B0_erasure_attrib_qwen7b.json --tag rjudge_qwen7b \
      --out outputs/vgap_mech/B1_ablation_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
from vgap_common import JUDGE_PROMPT, middle_out, parse_risk, control_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--b0", default="outputs/vgap_mech/B0_erasure_attrib_qwen7b.json")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/B1_ablation_qwen7b.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    b0 = json.load(open(args.b0))
    dsep_a = np.array(b0["dsep_attn"]); dsep_m = np.array(b0["dsep_mlp"])
    n_layers = b0["n_layers"]
    branches = [("attn", l, dsep_a[l]) for l in range(n_layers)] + [("mlp", l, dsep_m[l]) for l in range(n_layers)]
    erasers = [(b[0], b[1]) for b in sorted(branches, key=lambda z: z[2])[:args.topk]]        # most negative
    writers = [(b[0], b[1]) for b in sorted(branches, key=lambda z: -z[2])[:args.topk]]        # most positive
    rng = np.random.default_rng(args.seed)
    pool = [(t, l) for t in ("attn", "mlp") for l in range(n_layers) if (t, l) not in erasers]
    randoms = [tuple(pool[i]) for i in rng.choice(len(pool), args.topk, replace=False)]
    print(f"erasers={erasers}\nwriters(noneraser control)={writers}\nrandom control={randoms}", flush=True)

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    y_all = np.array([int(r.label) for r in recs])
    # balanced subsample for the (generation-heavy) B1
    idx = np.arange(len(recs)); rng.shuffle(idx)
    idx = idx[: args.n]
    recs = [recs[i] for i in idx]
    y = y_all[idx]
    texts = [r.flat_text() for r in recs]
    y_ctrl = np.array([control_label(t) for t in texts])
    print(f"{args.tag}: {len(recs)} records ({int(y.sum())} attack / {int(y_ctrl.sum())} URL)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map=os.environ.get("NSCC_DEVICE_MAP", "cuda")).eval()
    lm_head = model.get_output_embeddings()
    core = model.model if hasattr(model, "model") else model
    if getattr(core, "language_model", None) is not None:
        core = core.language_model
    layers = core.layers

    dt, dv = [], []
    for t_str, tid in tok.get_vocab().items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            dt.append(tid); dv.append(int(s))
    dt = torch.tensor(dt, device=model.device); dv = torch.tensor(dv, device=model.device, dtype=torch.float32)

    def exp_number(logits_row):
        p = torch.softmax(logits_row.float(), -1)[dt]
        return float((p / (p.sum() + 1e-8) * dv).sum())

    def build_enc(ctx):
        ids = tok.encode(ctx, add_special_tokens=False)
        ctx = tok.decode(middle_out(ids, args.max_ctx), skip_special_tokens=True)
        msg = [{"role": "user", "content": JUDGE_PROMPT.format(ctx=ctx)}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        if tok.chat_template:
            try:
                return tok.apply_chat_template(msg, enable_thinking=False, **kw)
            except TypeError:
                return tok.apply_chat_template(msg, **kw)
        return tok(JUDGE_PROMPT.format(ctx=ctx), return_tensors="pt")

    # ---- pass 1: benign-class mean of each candidate branch output at last position ----
    cands = list(set(erasers) | set(writers) | set(randoms))
    sums = {c: None for c in cands}; cnt = 0
    cap = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[name] = o[0, -1].detach().float()
        return hook
    hcap = []
    for (t, l) in cands:
        mod = layers[l].self_attn if t == "attn" else layers[l].mlp
        hcap.append(mod.register_forward_hook(mk(f"{t}{l}")))
    with torch.no_grad():
        for i, tx in enumerate(texts):
            if y[i] != 0:
                continue
            cap.clear(); enc = build_enc(tx).to(model.device)
            model(**enc, use_cache=False)
            for c in cands:
                v = cap[f"{c[0]}{c[1]}"]
                sums[c] = v.clone() if sums[c] is None else sums[c] + v
            cnt += 1
    for h in hcap:
        h.remove()
    bmean = {c: (sums[c] / max(1, cnt)).to(torch.bfloat16) for c in cands}
    print(f"benign-mean computed over {cnt} benign records", flush=True)

    # ---- ablation hooks: replace target branch output at the last position (prefill only) ----
    active = {"targets": []}
    def mk_abl(name, c):
        def hook(mod, inp, out):
            if c not in active["targets"]:
                return out
            o = out[0] if isinstance(out, tuple) else out
            if o.shape[1] > 1:                      # prefill (seq>1): the gen/score position
                o[0, -1] = bmean[c]
            return (o,) + out[1:] if isinstance(out, tuple) else o
        return hook
    habl = []
    for (t, l) in cands:
        mod = layers[l].self_attn if t == "attn" else layers[l].mlp
        habl.append(mod.register_forward_hook(mk_abl(f"{t}{l}", (t, l))))

    conditions = {"none": [], "eraser": erasers, "random": randoms, "noneraser": writers}
    res = {"tag": args.tag, "model": os.path.basename(args.model), "n": len(recs),
           "topk": args.topk, "erasers": erasers, "writers": writers, "randoms": randoms,
           "conditions": {}}
    with torch.no_grad():
        for cname, targets in conditions.items():
            active["targets"] = targets
            step0, verbal = [], []
            for tx in texts:
                enc = build_enc(tx).to(model.device)
                o = model(**enc, use_cache=False)          # ablated forward -> readout
                step0.append(exp_number(o.logits[0, -1]))
                gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
                verbal.append(parse_risk(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)))
            step0 = np.array(step0); verbal = np.array(verbal)
            sa, salo, sahi = bootstrap_ci(roc_auc, step0, y, n_boot=1000)
            va, valo, vahi = bootstrap_ci(roc_auc, verbal, y, n_boot=1000)
            row = {"step0_auroc": [sa, salo, sahi], "verbal_auroc": [va, valo, vahi],
                   "frac_parsed": float(np.mean(verbal > 0))}
            if cname == "eraser":
                row["url_verbal_auroc"] = float(roc_auc(verbal, y_ctrl))  # side-effect: control must hold
            res["conditions"][cname] = row
            print(f"  [{cname:9s}] step0={sa:.3f}  verbal={va:.3f}  parsed={row['frac_parsed']:.2f}", flush=True)
    for h in habl:
        h.remove()

    base_v = res["conditions"]["none"]["verbal_auroc"][0]
    er_v = res["conditions"]["eraser"]["verbal_auroc"][0]
    rnd_v = res["conditions"]["random"]["verbal_auroc"][0]
    non_v = res["conditions"]["noneraser"]["verbal_auroc"][0]
    base_s = res["conditions"]["none"]["step0_auroc"][0]
    er_s = res["conditions"]["eraser"]["step0_auroc"][0]
    res["verbal_recovery"] = er_v - base_v
    res["readout_recovery"] = er_s - base_s
    res["beats_controls"] = bool(er_v > rnd_v and er_v > non_v)
    res["gate_headline"] = bool((er_v - base_v) >= 0.15 and res["beats_controls"])
    res["gate_partial_readout_only"] = bool((er_s - base_s) >= 0.15 and (er_v - base_v) < 0.15)
    print(f"\nverbal: none={base_v:.3f} -> eraser={er_v:.3f} (recovery {er_v-base_v:+.3f}); "
          f"controls random={rnd_v:.3f} noneraser={non_v:.3f}")
    print(f"readout(step0): none={base_s:.3f} -> eraser={er_s:.3f} (recovery {er_s-base_s:+.3f})")
    print(f"HEADLINE gate (verbal +>=0.15 & beats both controls): {res['gate_headline']}  | "
          f"partial(readout-only): {res['gate_partial_readout_only']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
