"""G2: causal封顶 -- surgically remove the cancellation, does the SILENT model SPEAK?

From G0: the injection signal is present in components' contributions to the output readout
(pos-only AUROC ~0.6-0.75, null-checked) but a set of negative-discriminative components cancels
it to chance. G2 removes ONLY that cancellation contribution and asks whether the model's ACTUAL
generated verbal judgment recovers -- the strongest possible claim ("we made it speak").

Surgical intervention (fixes B1's failure: correct DLA-located targets + minimal disturbance):
for each cancelling component c, at the last position, set its projection on the readout direction
gd_hat to the BENIGN-class mean projection, leaving all orthogonal directions untouched:
  o_c' = o_c + (benign_mean_proj_c - <o_c, gd_hat>) * gd_hat
This removes exactly the component's contribution to the risk readout, nothing else.

Conditions: none | patch_neg (remove cancellation) | patch_random (count-matched) | patch_pos
(remove the signal-carrying components, should HURT). Metrics: Step0 readout AUROC + VERBAL AUROC
(actual greedy generation, parse_risk). Recovery = verbal(patch_neg) - verbal(none), must beat
controls. Component sets are chosen on a held-out split (no selection overfitting of the effect).

  CUDA_VISIBLE_DEVICES=0 python scripts/run_g2_patch.py --model Qwen/Qwen2.5-7B-Instruct \
      --tag rjudge_qwen7b --out outputs/vgap_mech/G2_patch_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
from vgap_common import JUDGE_PROMPT, middle_out, parse_risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/G2_patch_qwen7b.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(recs)); rng.shuffle(idx)
    if args.n:
        idx = idx[: args.n]
    recs = [recs[i] for i in idx]
    y = np.array([int(r.label) for r in recs])
    texts = [r.flat_text() for r in recs]
    # held-out split: select components on SEL, evaluate the intervention on EVAL
    sel_mask = np.zeros(len(y), bool); sel_mask[rng.choice(len(y), len(y) // 2, replace=False)] = True
    print(f"{args.tag}: {len(recs)} records ({int(y.sum())} attack); select on {sel_mask.sum()}, eval on {(~sel_mask).sum()}", flush=True)

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
    layers = core.layers; n_layers = len(layers)
    final_norm = getattr(core, "norm", None) or getattr(core, "final_layernorm", None)
    is_gemma = "gemma" in type(final_norm).__name__.lower()
    gain_w = (1.0 + final_norm.weight.detach().float()) if is_gemma else final_norm.weight.detach().float()

    W_U = lm_head.weight.detach()
    dt, dv = [], []
    for t_str, tid in tok.get_vocab().items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            dt.append(tid); dv.append(int(s))
    dt = torch.tensor(dt, device=W_U.device); dv = torch.tensor(dv, device=W_U.device, dtype=torch.float32)
    d = ((dv - dv.mean())[:, None] * W_U[dt].float()).sum(0); d = d / (d.norm() + 1e-8)
    gd = gain_w * d
    gd_hat = (gd / gd.norm()).to(torch.bfloat16)                # unit readout direction (hidden space)

    comp_mods = [("emb", None)] + [("a", l) for l in range(n_layers)] + [("m", l) for l in range(n_layers)]
    def mod_of(kind, l):
        return layers[l].self_attn if kind == "a" else layers[l].mlp
    cap = {}
    def mk(key):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[key] = o[0, -1].detach().float()
        return hook
    hcap = []
    for kind, l in comp_mods:
        if kind == "emb":
            continue
        hcap.append(mod_of(kind, l).register_forward_hook(mk(f"{kind}{l}")))

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

    # ---- pass 1 (no patch): per-comp readout contribution on SEL to pick neg/pos; benign-mean proj ----
    keys = [f"{k}{l}" for k, l in comp_mods if k != "emb"]
    contrib = {k: [] for k in keys}; proj = {k: [] for k in keys}
    with torch.no_grad():
        for i, t in enumerate(texts):
            cap.clear(); enc = build_enc(t).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hL = out.hidden_states[0][0, -1].float()
            for k in keys:
                hL = hL + cap[k]
            sigma = torch.sqrt((hL * hL).mean() + 1e-6)
            for k in keys:
                contrib[k].append(float((cap[k] @ gd) / sigma))
                proj[k].append(float(cap[k] @ gd_hat.float()))
            if i % 100 == 0:
                print(f"  scan {i}/{len(texts)}", flush=True)
    contrib = {k: np.array(v) for k, v in contrib.items()}
    proj = {k: np.array(v) for k, v in proj.items()}
    sel = sel_mask
    ca = {k: roc_auc(contrib[k][sel], y[sel]) for k in keys}
    neg_keys = [k for k in keys if ca[k] <= 0.45]
    pos_keys = [k for k in keys if ca[k] >= 0.55]
    rng2 = np.random.default_rng(1)
    rand_keys = [keys[i] for i in rng2.choice(len(keys), len(neg_keys), replace=False)]
    # benign-mean projection per component (on SEL benign)
    bmp = {k: float(proj[k][sel & (y == 0)].mean()) for k in keys}
    print(f"neg(cancel)={len(neg_keys)} pos(signal)={len(pos_keys)} random={len(rand_keys)}", flush=True)

    # ---- surgical patch hooks: set target components' gd_hat projection to benign-mean ----
    active = {"targets": []}
    ghat = gd_hat
    def mk_patch(key):
        def hook(mod, inp, out):
            if key not in active["targets"]:
                return out
            o = out[0] if isinstance(out, tuple) else out
            if o.shape[1] > 1:                                  # prefill: the score position
                v = o[0, -1]
                cur = torch.dot(v, ghat)
                o[0, -1] = v + (bmp[key] - float(cur)) * ghat
            return (o,) + out[1:] if isinstance(out, tuple) else o
        return hook
    hpatch = []
    for kind, l in comp_mods:
        if kind == "emb":
            continue
        hpatch.append(mod_of(kind, l).register_forward_hook(mk_patch(f"{kind}{l}")))

    ev = ~sel_mask
    conditions = {"none": [], "patch_neg": neg_keys, "patch_random": rand_keys, "patch_pos": pos_keys}
    res = {"tag": args.tag, "model": os.path.basename(args.model), "n_eval": int(ev.sum()),
           "n_neg": len(neg_keys), "n_pos": len(pos_keys), "conditions": {}}
    with torch.no_grad():
        for cname, targets in conditions.items():
            active["targets"] = set(targets)
            step0, verbal = [], []
            for i, t in enumerate(texts):
                if not ev[i]:
                    continue
                enc = build_enc(t).to(model.device)
                o = model(**enc, output_hidden_states=False, use_cache=False)
                p = torch.softmax(o.logits[0, -1].float(), -1)[dt]
                step0.append(float((p / (p.sum() + 1e-8) * dv).sum()))
                gen = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
                verbal.append(parse_risk(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)))
            step0 = np.array(step0); verbal = np.array(verbal); yev = y[ev]
            sa = float(roc_auc(step0, yev)); va, vlo, vhi = bootstrap_ci(roc_auc, verbal, yev, n_boot=1000)
            res["conditions"][cname] = {"step0_auroc": sa, "verbal_auroc": [va, vlo, vhi],
                                        "frac_parsed": float(np.mean(verbal > 0))}
            print(f"  [{cname:12s}] step0={sa:.3f}  verbal={va:.3f} [{vlo:.2f},{vhi:.2f}]", flush=True)
    for h in hcap + hpatch:
        h.remove()

    bv = res["conditions"]["none"]["verbal_auroc"][0]; nv = res["conditions"]["patch_neg"]["verbal_auroc"][0]
    rv = res["conditions"]["patch_random"]["verbal_auroc"][0]; pv = res["conditions"]["patch_pos"]["verbal_auroc"][0]
    bs = res["conditions"]["none"]["step0_auroc"]; ns = res["conditions"]["patch_neg"]["step0_auroc"]
    res["verbal_recovery"] = nv - bv; res["readout_recovery"] = ns - bs
    res["beats_controls"] = bool(nv > rv + 0.03 and nv > bv)
    res["causal_headline"] = bool((nv - bv) >= 0.10 and res["beats_controls"])
    print(f"\nVERBAL: none={bv:.3f} -> patch_neg={nv:.3f} (recovery {nv-bv:+.3f}); controls random={rv:.3f} pos={pv:.3f}")
    print(f"READOUT(step0): none={bs:.3f} -> patch_neg={ns:.3f} (recovery {ns-bs:+.3f})")
    print(f"CAUSAL HEADLINE (verbal +>=0.10 & beats controls): {res['causal_headline']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
