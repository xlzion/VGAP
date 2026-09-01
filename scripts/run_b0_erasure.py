"""B0: erasure attribution (observational, no intervention).

Upgrades the S1 finding "the late-layer logit-lens for injection collapses to chance" into
"WHICH components write the readable injection signal DOWN". Uses the exact additivity of the
residual stream: h_{l+1} = h_l + attn_out_l + mlp_out_l (last position). Projecting each branch
output onto a frozen READOUT DIRECTION d (the direction in hidden space that raises the emitted
risk number) gives each component's signed push on the model's own readout. A branch whose
attack-minus-benign push is strongly NEGATIVE is an eraser.

Readout direction (pre-registered): d = normalize( sum_t (val_t - mean_val) * W_U[t] ), t over
the frozen pure-digit token set (0-100), val_t its integer value. c_l = <h_l, d> (raw residual;
additive). Consistency check: per-layer AUROC(c_l) must reproduce the S1 curve-B shape
(mid ~high, late collapse) else the direction is void and B0 stops.

Also runs the SAME decomposition for the URL control concept -- expected: the control's readable
component is NOT erased (erasure is injection-specific).

  CUDA_VISIBLE_DEVICES=1 python scripts/run_b0_erasure.py \
      --model Qwen/Qwen2.5-7B-Instruct --tag rjudge_qwen7b \
      --out outputs/vgap_mech/B0_erasure_attrib_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
from vgap_common import JUDGE_PROMPT, middle_out, control_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/B0_erasure_attrib_qwen7b.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    if args.n:
        recs = recs[: args.n]
    y = np.array([int(r.label) for r in recs])
    texts = [r.flat_text() for r in recs]
    y_ctrl = np.array([control_label(t) for t in texts])
    print(f"{args.tag}: {len(recs)} records ({int(y.sum())} attack / {int(y_ctrl.sum())} URL) | "
          f"{os.path.basename(args.model)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map=os.environ.get("NSCC_DEVICE_MAP", "cuda")).eval()

    # locate LM core, decoder layers, unembedding
    lm_head = model.get_output_embeddings()
    core = model.model if hasattr(model, "model") else model
    if getattr(core, "language_model", None) is not None:
        core = core.language_model
    layers = core.layers
    n_layers = len(layers)

    # frozen pure-digit token set (0-100) and readout direction d
    W_U = lm_head.weight.detach()                       # (V, H)
    digit_tok, digit_val = [], []
    for t_str, tid in tok.get_vocab().items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            digit_tok.append(tid); digit_val.append(int(s))
    dt = torch.tensor(digit_tok, device=W_U.device)
    dv = torch.tensor(digit_val, device=W_U.device, dtype=torch.float32)
    w = (dv - dv.mean())                                # (K,)
    d = (w[:, None] * W_U[dt].float()).sum(0)           # (H,)
    d = d / (d.norm() + 1e-8)
    print(f"readout direction d over {len(digit_tok)} digit tokens; |d|={d.norm():.3f}", flush=True)

    # hooks to capture attn-branch and mlp-branch outputs at the last position
    cap = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[name] = o[0, -1].detach().float()        # (H,) last position
        return hook
    handles = []
    for li, lyr in enumerate(layers):
        handles.append(lyr.self_attn.register_forward_hook(mk(f"a{li}")))
        handles.append(lyr.mlp.register_forward_hook(mk(f"m{li}")))

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

    C = np.zeros((len(texts), n_layers + 1))            # <h_l, d> per record per layer
    A = np.zeros((len(texts), n_layers))                # <attn_out_l, d>
    M = np.zeros((len(texts), n_layers))                # <mlp_out_l, d>
    with torch.no_grad():
        for i, t in enumerate(texts):
            cap.clear()
            enc = build_enc(t).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states                       # (L+1) of (1, seq, H)
            for l in range(n_layers + 1):
                C[i, l] = float(hs[l][0, -1].float() @ d)
            for l in range(n_layers):
                A[i, l] = float(cap[f"a{l}"] @ d)
                M[i, l] = float(cap[f"m{l}"] @ d)
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}", flush=True)
    for h in handles:
        h.remove()

    # additivity sanity: C[:,l+1] ~ C[:,l] + A[:,l] + M[:,l].
    # NOTE: transformers 5.x returns hidden_states[-1] POST-final-norm, so the LAST transition
    # is not additive (norm, not a bug). Branch captures A/M are correct for every layer incl.
    # the last; we check additivity over layers 0..L-2 where hidden_states are raw residual.
    resid = C[:, 1:n_layers] - C[:, :n_layers - 1] - A[:, :n_layers - 1] - M[:, :n_layers - 1]
    add_err = float(np.abs(resid).mean())

    # consistency check: per-layer AUROC(c_l) reproduces S1 curve B (mid high, late collapse)
    c_auroc = [float(roc_auc(C[:, l], y)) for l in range(n_layers + 1)]

    # SCALE-INVARIANT class-conditional separation per branch (Cohen's-d effect size):
    # raw mean-difference grows with residual magnitude (deep layers) and mislocates the eraser
    # to late high-norm layers; normalizing by the pooled std makes layers/hosts comparable and
    # puts the eraser where discriminability is actually lost.
    def sep(x):  # raw mean_attack - mean_benign
        return float(x[y == 1].mean() - x[y == 0].mean())
    def dsep_es(x):  # effect size = (mean_atk - mean_ben) / pooled_std
        s = np.sqrt(0.5 * (x[y == 1].var() + x[y == 0].var())) + 1e-8
        return float((x[y == 1].mean() - x[y == 0].mean()) / s)
    dsep_a_raw = np.array([sep(A[:, l]) for l in range(n_layers)])
    dsep_m_raw = np.array([sep(M[:, l]) for l in range(n_layers)])
    dsep_a = np.array([dsep_es(A[:, l]) for l in range(n_layers)])   # effect-size (used for ranking)
    dsep_m = np.array([dsep_es(M[:, l]) for l in range(n_layers)])

    # eraser ranking (most negative Δsep). concentration kappa(k) over the late half.
    late0 = n_layers // 2
    branches = ([("attn", l, dsep_a[l]) for l in range(n_layers)]
                + [("mlp", l, dsep_m[l]) for l in range(n_layers)])
    erasers = sorted(branches, key=lambda z: z[2])       # ascending: most negative first
    late_erasers = [b for b in erasers if b[1] >= late0 and b[2] < 0]
    total_neg = -sum(b[2] for b in late_erasers) or 1e-8
    def kappa(k):
        return float(-sum(b[2] for b in late_erasers[:k]) / total_neg)
    top = [{"branch": b[0], "layer": b[1], "dsep": b[2]} for b in erasers[:8]]

    # control concept: same branch projections, effect-size separation by URL label
    def dsep_es_ctrl(x):
        s = np.sqrt(0.5 * (x[y_ctrl == 1].var() + x[y_ctrl == 0].var())) + 1e-8
        return float((x[y_ctrl == 1].mean() - x[y_ctrl == 0].mean()) / s)
    dsep_a_ctrl = np.array([dsep_es_ctrl(A[:, l]) for l in range(n_layers)])
    dsep_m_ctrl = np.array([dsep_es_ctrl(M[:, l]) for l in range(n_layers)])
    late_erase_inj = float(sum(min(0, dsep_a[l]) + min(0, dsep_m[l]) for l in range(late0, n_layers)))
    late_erase_ctrl = float(sum(min(0, dsep_a_ctrl[l]) + min(0, dsep_m_ctrl[l]) for l in range(late0, n_layers)))

    concentrated = bool(kappa(3) >= 0.60)
    diffuse = bool(kappa(10) < 0.50)
    res = {"tag": args.tag, "model": os.path.basename(args.model), "n": len(recs),
           "n_layers": n_layers, "n_digit_tokens": len(digit_tok),
           "additivity_abs_err": add_err,
           "c_auroc_per_layer": c_auroc,
           "dsep_attn": dsep_a.tolist(), "dsep_mlp": dsep_m.tolist(),
           "dsep_attn_raw": dsep_a_raw.tolist(), "dsep_mlp_raw": dsep_m_raw.tolist(),
           "ranking_metric": "effect_size",
           "top_erasers": top, "kappa3": kappa(3), "kappa10": kappa(10),
           "late_erasure_injection": late_erase_inj, "late_erasure_control_url": late_erase_ctrl,
           "verdict": "concentrated" if concentrated else ("diffuse" if diffuse else "intermediate")}
    print(f"\nadditivity |err|={add_err:.4f} (should be ~0)")
    print(f"c_auroc: L0={c_auroc[0]:.3f} mid={c_auroc[n_layers//2]:.3f} last={c_auroc[-1]:.3f} "
          f"(must mirror S1 curve B: mid high, last low)")
    print("top erasers (most negative attack-benign push on readout):")
    for b in top:
        print(f"   {b['branch']} L{b['layer']:2d}  dsep={b['dsep']:+.3f}")
    print(f"kappa(3)={kappa(3):.2f}  kappa(10)={kappa(10):.2f} -> {res['verdict']}")
    print(f"late erasure  injection={late_erase_inj:+.3f}  vs  URL-control={late_erase_ctrl:+.3f} "
          f"(control should erase far less -> injection-specific)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
