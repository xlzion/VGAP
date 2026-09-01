"""G1: is the cancellation SIGNAL-DRIVEN (active suppression) or a fixed offset?

From G0 the injection signal is present in components' contributions to the readout (pos-only
AUROC ~0.62-0.67, null-checked) but the total readout s is chance -- cancellation. G1 asks WHAT
the cancellation reads. Using the frozen-norm DLA cumulative readout trajectory
  c_l = emb_contrib + sum_{k<=l} (attn_k + mlp_k)_contrib     (c_final = total s ~ chance)
we locate the peak layer L* (max AUROC(c_l), the point of maximal output-readability) and ask:
does the mid-peak signal c_peak propagate to the final readout c_final?

  active/signal-driven suppression  => WITHIN ATTACKS c_peak is DECOUPLED from c_final
       (corr ~ 0 / slope ~ 0): however strong the mid signal, it is removed by output. And this
       decoupling is SELECTIVE: benign c_peak->c_final couples more (nothing to remove).
  fixed offset (not signal-driven)  => c_final = c_peak - const, so corr(c_peak,c_final) ~ 1.

Controls: within-benign coupling; label-permutation null; bootstrap CI on the within-attack corr.
Observational (arithmetic on stored contributions), NO intervention.

  CUDA_VISIBLE_DEVICES=0 python scripts/run_g1_suppression.py --model Qwen/Qwen2.5-7B-Instruct \
      --tag rjudge_qwen7b --out outputs/vgap_mech/G1_suppression_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc
from vgap_common import JUDGE_PROMPT, middle_out


def corr(a, b):
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/G1_suppression_qwen7b.json")
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
    gain_w = (1.0 + final_norm.weight.detach().float()) if is_gemma else final_norm.weight.detach().float()
    eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))

    W_U = lm_head.weight.detach()
    dt, dv = [], []
    for t_str, tid in tok.get_vocab().items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            dt.append(tid); dv.append(int(s))
    dt = torch.tensor(dt, device=W_U.device); dv = torch.tensor(dv, device=W_U.device, dtype=torch.float32)
    d = ((dv - dv.mean())[:, None] * W_U[dt].float()).sum(0); d = d / (d.norm() + 1e-8)
    gd = gain_w * d

    cap = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[name] = o[0, -1].detach().float()
        return hook
    for li, lyr in enumerate(layers):
        lyr.self_attn.register_forward_hook(mk(f"a{li}"))
        lyr.mlp.register_forward_hook(mk(f"m{li}"))

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

    # cumulative frozen-norm readout trajectory c_l per record (c[:,0]=emb, c[:,l]=through layer l-1)
    Cc = np.zeros((len(texts), n_layers + 1))
    with torch.no_grad():
        for i, t in enumerate(texts):
            cap.clear(); enc = build_enc(t).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            emb = out.hidden_states[0][0, -1].float()
            branches = [cap[f"a{l}"] + cap[f"m{l}"] for l in range(n_layers)]
            hL = emb.clone()
            for b in branches:
                hL = hL + b
            sigma = torch.sqrt((hL * hL).mean() + eps)
            run = emb.clone(); Cc[i, 0] = float((run @ gd) / sigma)
            for l in range(n_layers):
                run = run + branches[l]
                Cc[i, l + 1] = float((run @ gd) / sigma)
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}", flush=True)

    layer_auroc = np.array([roc_auc(Cc[:, l], y) for l in range(n_layers + 1)])
    L_peak = int(np.argmax(layer_auroc))
    c_peak = Cc[:, L_peak]; c_final = Cc[:, -1]
    atk = y == 1; ben = y == 0

    # within-attack coupling of mid-peak signal to final readout (low = active removal)
    r_atk = corr(c_peak[atk], c_final[atk])
    r_ben = corr(c_peak[ben], c_final[ben])
    # slope of c_final ~ c_peak within attacks
    slope_atk = float(np.polyfit(c_peak[atk], c_final[atk], 1)[0]) if atk.sum() > 3 else float("nan")
    slope_ben = float(np.polyfit(c_peak[ben], c_final[ben], 1)[0]) if ben.sum() > 3 else float("nan")

    # bootstrap CI on within-attack corr
    rng = np.random.default_rng(0); idx_a = np.where(atk)[0]; boots = []
    for _ in range(2000):
        s = rng.choice(idx_a, len(idx_a), replace=True)
        boots.append(corr(c_peak[s], c_final[s]))
    boots = [b for b in boots if not np.isnan(b)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"),) * 2

    # label-permutation null: within-"attack" coupling under shuffled labels
    null_r = []
    for _ in range(50):
        yp = rng.permutation(y)
        null_r.append(corr(c_peak[yp == 1], c_final[yp == 1]))
    null_r = [b for b in null_r if not np.isnan(b)]
    null_mean = float(np.mean(null_r)); null_lo, null_hi = float(np.percentile(null_r, 2.5)), float(np.percentile(null_r, 97.5))

    # verdict: signal-driven active suppression if attack coupling is LOW and CLEARLY below benign
    active = bool((abs(r_atk) < 0.35) and (r_ben - r_atk >= 0.20))
    res = {"tag": args.tag, "model": os.path.basename(args.model), "n": len(recs),
           "L_peak": L_peak, "L_peak_frac": L_peak / n_layers, "peak_auroc": float(layer_auroc[L_peak]),
           "final_auroc": float(layer_auroc[-1]),
           "corr_peak_final_attack": r_atk, "corr_peak_final_benign": r_ben,
           "corr_attack_ci": list(ci), "slope_attack": slope_atk, "slope_benign": slope_ben,
           "null_corr_mean": null_mean, "null_corr_ci": [null_lo, null_hi],
           "verdict_signal_driven_suppression": active}
    print(f"\npeak L{L_peak}({L_peak/n_layers:.0%}) AUROC={layer_auroc[L_peak]:.3f} -> final AUROC={layer_auroc[-1]:.3f}")
    print(f"within-ATTACK  corr(c_peak,c_final)={r_atk:.3f} CI[{ci[0]:.2f},{ci[1]:.2f}]  slope={slope_atk:.2f}")
    print(f"within-BENIGN  corr(c_peak,c_final)={r_ben:.3f}  slope={slope_ben:.2f}")
    print(f"NULL (shuffled labels) corr mean={null_mean:.3f} CI[{null_lo:.2f},{null_hi:.2f}]")
    print(f"VERDICT signal-driven active suppression (attack decoupled & below benign): {active}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
