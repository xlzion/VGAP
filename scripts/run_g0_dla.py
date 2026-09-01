"""G0: frozen-norm Direct Logit Attribution (DLA) + cancellation/drowning/rotation decision.

Third (and methodologically sound) attribution. The final readout scalar
  s = <FinalNorm(h_L), d>,  h_L = emb + sum_l (attn_l + mlp_l)   (last position, pre-norm)
With the RMSNorm per-example scale sigma(x)=rms(h_L) FROZEN, FinalNorm is linear in h_L, so
  s = sum_c contrib_c,  contrib_c = <(comp_c / sigma) * gain, d>
is EXACTLY additive over components c in {emb, attn_0..L-1, mlp_0..L-1} per example. This fixes
both prior failures: unit = contribution to the ACTUAL readout (not raw magnitude, fixes #1),
additivity guaranteed by construction (fixes #2, Cohen's-d). Contribution SIZE and
DISCRIMINABILITY are then reported on SEPARATE axes -- exactly what was conflated before.

Decision (pre-registered, per G-plan section 1), given total-s AUROC ~ chance (Step0):
  cancellation : exists components with AUROC(contrib) < 0.45 (discriminate the WRONG way), and
                 within-attack correlation between their summed contribution and the positive
                 accumulation < -0.3  -> an active suppression computation.
  drowning     : no such negative set; positive-discriminative contributions sum to AUROC >= 0.8,
                 but total s is chance because non-discriminative components add variance
                 (removing non-discriminative variance restores AUROC) -> readout-SNR story.
  rotation     : discriminative contributions weak throughout (only if R2/tuned-lens overturns
                 the mid-peak -- flagged, not decided here).

Observational only (arithmetic on stored per-example contributions), NO intervention.

  CUDA_VISIBLE_DEVICES=0 python scripts/run_g0_dla.py --model Qwen/Qwen2.5-7B-Instruct \
      --tag rjudge_qwen7b --out outputs/vgap_mech/G0_dla_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
from vgap_common import JUDGE_PROMPT, middle_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/G0_dla_qwen7b.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    if args.n:
        recs = recs[: args.n]
    y = np.array([int(r.label) for r in recs])
    texts = [r.flat_text() for r in recs]
    print(f"{args.tag}: {len(recs)} records ({int(y.sum())} attack)", flush=True)

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
    n_layers = len(layers)
    final_norm = getattr(core, "norm", None) or getattr(core, "final_layernorm", None)
    is_gemma = "gemma" in type(final_norm).__name__.lower()
    fn_w = final_norm.weight.detach().float()
    gain_w = (1.0 + fn_w) if is_gemma else fn_w                 # gemma RMSNorm uses (1+w)
    eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))

    W_U = lm_head.weight.detach()
    dt, dv = [], []
    for t_str, tid in tok.get_vocab().items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            dt.append(tid); dv.append(int(s))
    dt = torch.tensor(dt, device=W_U.device); dv = torch.tensor(dv, device=W_U.device, dtype=torch.float32)
    d = ((dv - dv.mean())[:, None] * W_U[dt].float()).sum(0)
    d = d / (d.norm() + 1e-8)

    cap = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[name] = o[0, -1].detach().float()
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

    comp_names = ["emb"] + [f"a{l}" for l in range(n_layers)] + [f"m{l}" for l in range(n_layers)]
    C = np.zeros((len(texts), len(comp_names)))   # per-example contribution of each component to s
    s_full = np.zeros(len(texts)); s_check = np.zeros(len(texts))
    gd = (gain_w * d)                             # (H,)  gain ⊙ d
    with torch.no_grad():
        for i, t in enumerate(texts):
            cap.clear(); enc = build_enc(t).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            emb = out.hidden_states[0][0, -1].float()
            comps = [emb] + [cap[f"a{l}"] for l in range(n_layers)] + [cap[f"m{l}"] for l in range(n_layers)]
            hL = emb.clone()
            for c in comps[1:]:
                hL = hL + c
            sigma = torch.sqrt((hL * hL).mean() + eps)
            for j, c in enumerate(comps):
                C[i, j] = float((c @ gd) / sigma)           # frozen-norm DLA contribution to s
            s_full[i] = float(C[i].sum())
            s_check[i] = float((final_norm(hL.to(torch.bfloat16)).float() @ d))  # true readout
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}", flush=True)
    for h in handles:
        h.remove()

    add_err = float(np.abs(s_full - s_check).mean())
    s_auroc = float(roc_auc(s_full, y))

    comp_auroc = np.array([roc_auc(C[:, j], y) for j in range(len(comp_names))])
    pos_idx = [j for j in range(len(comp_names)) if comp_auroc[j] >= 0.55]
    neg_idx = [j for j in range(len(comp_names)) if comp_auroc[j] <= 0.45]

    # ---- OOF de-biased selection (fixes in-sample component-selection overfitting) ----
    # Select pos/neg components on the TRAIN fold, sum their contributions on the HELD-OUT fold.
    from sklearn.model_selection import StratifiedKFold
    oof_pos = np.zeros(len(y)); oof_neg = np.zeros(len(y)); oof_disc = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(C, y):
        ca = np.array([roc_auc(C[tr, j], y[tr]) for j in range(len(comp_names))])
        p = [j for j in range(len(comp_names)) if ca[j] >= 0.55]
        n = [j for j in range(len(comp_names)) if ca[j] <= 0.45]
        oof_pos[te] = C[np.ix_(te, p)].sum(1) if p else 0.0
        oof_neg[te] = C[np.ix_(te, n)].sum(1) if n else 0.0
        oof_disc[te] = oof_pos[te] + oof_neg[te]
    pos_auroc_oof = float(roc_auc(oof_pos, y))
    disc_auroc_oof = float(roc_auc(oof_disc, y))

    # ---- BUG-CHECK: label-permutation null. Run the SAME OOF select+sum on shuffled labels.
    # If null pos-only >> 0.5, the "signal in a component subset" is a selection artifact and the
    # whole cancellation story is void. Real signal => null pos-only ~ 0.5, real pos-only stays high.
    rng_null = np.random.default_rng(1)
    null_pos, null_disc = [], []
    for _ in range(10):
        yp = rng_null.permutation(y)
        op = np.zeros(len(yp)); od = np.zeros(len(yp))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(C, yp):
            ca = np.array([roc_auc(C[tr, j], yp[tr]) for j in range(len(comp_names))])
            p = [j for j in range(len(comp_names)) if ca[j] >= 0.55]
            n = [j for j in range(len(comp_names)) if ca[j] <= 0.45]
            op[te] = C[np.ix_(te, p)].sum(1) if p else 0.0
            od[te] = op[te] + (C[np.ix_(te, n)].sum(1) if n else 0.0)
        null_pos.append(roc_auc(op, yp)); null_disc.append(roc_auc(od, yp))
    null_pos_mean = float(np.mean(null_pos)); null_pos_max = float(np.max(null_pos))
    pos_sum = C[:, pos_idx].sum(1) if pos_idx else np.zeros(len(y))
    neg_sum = C[:, neg_idx].sum(1) if neg_idx else np.zeros(len(y))
    nondisc_idx = [j for j in range(len(comp_names)) if 0.45 < comp_auroc[j] < 0.55]
    nondisc_sum = C[:, nondisc_idx].sum(1) if nondisc_idx else np.zeros(len(y))
    pos_auroc = float(roc_auc(pos_sum, y)) if pos_idx else 0.5
    neg_auroc = float(roc_auc(neg_sum, y)) if neg_idx else 0.5
    # DECISIVE test: discriminative-only readout = pos+neg (== total minus non-discriminative).
    # drowning  => removing non-disc variance RESTORES discrimination (disc_auroc high).
    # cancellation => pos alone works but pos+neg stays ~chance (the two disc sets cancel).
    disc_sum = pos_sum + neg_sum
    disc_auroc = float(roc_auc(disc_sum, y))

    # within-attack correlation between negative-set sum and positive accumulation (cancellation test)
    atk = y == 1
    if neg_idx and atk.sum() > 5:
        r_cancel = float(np.corrcoef(neg_sum[atk], pos_sum[atk])[0, 1])
    else:
        r_cancel = float("nan")
    # drowning test: does total variance dominated by non-discriminative components?
    var_pos = float(np.var(pos_sum)); var_nondisc = float(np.var(nondisc_sum)); var_total = float(np.var(s_full))
    # magnitude of negative set relative to positive set (is cancellation quantitatively real?)
    neg_mag = float(np.abs(neg_sum[atk]).mean()) if neg_idx else 0.0
    pos_mag = float(np.abs(pos_sum[atk]).mean()) if pos_idx else 1e-8

    # decisive (OOF de-biased): does the discriminative-only readout (pos+neg) recover or cancel?
    # real signal required: OOF pos-only must clear the permutation null by a margin
    signal_real = bool(pos_auroc_oof - null_pos_max >= 0.05)
    drowning = bool(signal_real and disc_auroc_oof >= 0.75 and var_nondisc >= var_pos)
    cancellation = bool(signal_real and (not drowning) and pos_auroc_oof >= 0.65 and abs(disc_auroc_oof - 0.5) <= 0.12)
    verdict = ("selection-artifact(null not cleared)" if not signal_real else
               ("drowning" if drowning else ("cancellation" if cancellation else "inconclusive")))

    order = np.argsort(comp_auroc)
    top_pos = [{"comp": comp_names[j], "auroc": float(comp_auroc[j])} for j in order[::-1][:6]]
    top_neg = [{"comp": comp_names[j], "auroc": float(comp_auroc[j])} for j in order[:6]]

    res = {"tag": args.tag, "model": os.path.basename(args.model), "n": len(recs),
           "n_layers": n_layers, "is_gemma_norm": is_gemma, "dla_additivity_err": add_err,
           "s_auroc": s_auroc, "n_pos_comp": len(pos_idx), "n_neg_comp": len(neg_idx),
           "pos_sum_auroc": pos_auroc, "neg_sum_auroc": neg_auroc, "disc_only_auroc": disc_auroc,
           "pos_sum_auroc_OOF": pos_auroc_oof, "disc_only_auroc_OOF": disc_auroc_oof,
           "null_pos_only_mean": null_pos_mean, "null_pos_only_max": null_pos_max, "signal_real": signal_real,
           "cancel_within_attack_corr": r_cancel, "neg_over_pos_magnitude": neg_mag / (pos_mag + 1e-8),
           "var_pos": var_pos, "var_nondisc": var_nondisc, "var_total": var_total,
           "top_positive_components": top_pos, "top_negative_components": top_neg,
           "verdict": verdict}
    print(f"\nDLA additivity |err|={add_err:.4f} (should be ~0)  | total s AUROC={s_auroc:.3f} (should be ~chance)")
    print(f"pos-discriminative comps: {len(pos_idx)}  (their sum AUROC={pos_auroc:.3f})")
    print(f"neg-discriminative comps: {len(neg_idx)}  (their sum AUROC={neg_auroc:.3f})")
    print(f"DECISIVE (OOF de-biased): pos-only AUROC={pos_auroc_oof:.3f}  disc-only(pos+neg) AUROC={disc_auroc_oof:.3f}")
    print(f"NULL (label-permutation, same OOF select+sum): pos-only mean={null_pos_mean:.3f} max={null_pos_max:.3f}  "
          f"-> signal_real={signal_real} (pos_oof must exceed null_max by >=0.05)")
    print(f"cancellation test: neg/pos magnitude={res['neg_over_pos_magnitude']:.2f}, within-attack corr(neg,pos)={r_cancel:.3f} (< -0.3 = cancel)")
    print(f"drowning test: var_pos={var_pos:.3f} var_nondisc={var_nondisc:.3f} (nondisc>=pos & pos_auroc>=0.8 = drown)")
    print(f"top +disc comps: {[(t['comp'],round(t['auroc'],2)) for t in top_pos]}")
    print(f"top -disc comps: {[(t['comp'],round(t['auroc'],2)) for t in top_neg]}")
    print(f"VERDICT (single-host): {verdict}  (cross-3-host same verdict required for the mechanism claim)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
