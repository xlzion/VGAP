"""R2: tuned-lens vs frozen-lens -- formally close 'rotation' and confirm cancellation.

Vanilla (frozen) logit-lens reads the RAW residual through the model's actual unembedding. Its
mid-peak (S1: 0.71-0.94 at ~40% depth) says the injection IS in the unembedding-readable subspace
mid-network -> rules out 'never-aligned / rotation'. But mid-layer frozen lens can be basis-
mismatched. A TUNED lens learns a per-layer affine map h_l -> final digit-logits (reproducing what
the model WOULD output), correcting the basis. If:
  - tuned mid still discriminates (peak) -> mid-peak is real, not a frozen-lens artifact (rotation OUT);
  - tuned LAST-layer ~ chance (like Step0) -> the model's own output removes it (cancellation confirmed).
We report both curves per layer (OOF, digit-logit ridge). Cheap: 10-dim targets.

  CUDA_VISIBLE_DEVICES=0 python scripts/run_r2_tunedlens.py --model Qwen/Qwen2.5-7B-Instruct \
      --tag rjudge_qwen7b --out outputs/vgap_mech/R2_tunedlens_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc
from vgap_common import JUDGE_PROMPT, middle_out
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/R2_tunedlens_qwen7b.json")
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
    final_norm = getattr(core, "norm", None) or getattr(core, "final_layernorm", None)
    is_gemma = "gemma" in type(final_norm).__name__.lower()
    softcap = getattr(model.config, "final_logit_softcapping", None)

    W_U = lm_head.weight.detach()
    dt, dv = [], []
    for t_str, tid in tok.get_vocab().items():
        s = t_str.replace("▁", " ").replace("Ġ", " ").strip()
        if s.isascii() and s.isdigit() and len(s) <= 3 and int(s) <= 100:
            dt.append(tid); dv.append(int(s))
    dt_t = torch.tensor(dt, device=W_U.device); dv_np = np.array(dv, float)
    d = ((torch.tensor(dv, device=W_U.device, dtype=torch.float32) - float(np.mean(dv)))[:, None]
         * W_U[dt_t].float()).sum(0)
    d = (d / (d.norm() + 1e-8))

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

    def frozen_norm(h):
        x = final_norm(h.to(torch.bfloat16))
        z = lm_head(x)
        if softcap:
            z = softcap * torch.tanh(z / softcap)
        return z

    def exp_num_from_digitlogits(dl):  # dl: (N,K) numpy digit logits -> expected number
        p = np.exp(dl - dl.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return (p * dv_np[None, :]).sum(1)

    n_hidden = None
    H_all = []                 # per record: (L+1, hidden) last-position hidden
    T_all = []                 # per record: (K,) final digit-logits (model's actual output target)
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = build_enc(t).to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states; n_hidden = len(hs)
            H_all.append(np.stack([hs[l][0, -1].float().cpu().numpy() for l in range(n_hidden)]))
            T_all.append(out.logits[0, -1][dt_t].float().cpu().numpy())
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}", flush=True)
    H = np.stack(H_all); T = np.stack(T_all)          # H:(N,L+1,d) T:(N,K)

    frozen_curve, tuned_curve = [], []
    with torch.no_grad():
        for l in range(n_hidden):
            Xl = H[:, l, :]
            # frozen lens readout (actual unembedding), expected number
            fl = []
            for j0 in range(0, len(Xl), 64):
                hb = torch.tensor(Xl[j0:j0 + 64], device=model.device, dtype=torch.bfloat16)
                z = frozen_norm(hb)[:, dt_t].float().cpu().numpy()
                fl.append(exp_num_from_digitlogits(z))
            frozen_s = np.concatenate(fl)
            frozen_curve.append(float(roc_auc(frozen_s, y)))
            # tuned lens: OOF ridge h_l -> final digit-logits, then expected number
            tuned_s = np.zeros(len(Xl))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(Xl):
                rg = Ridge(alpha=1000.0).fit(Xl[tr], T[tr])
                tuned_s[te] = exp_num_from_digitlogits(rg.predict(Xl[te]))
            tuned_curve.append(float(roc_auc(tuned_s, y)))
            if l in (0, n_hidden // 3, n_hidden // 2, 2 * n_hidden // 3, n_hidden - 1):
                print(f"  L{l:2d}: frozen={frozen_curve[-1]:.3f}  tuned={tuned_curve[-1]:.3f}", flush=True)

    fp = int(np.argmax(frozen_curve)); tp = int(np.argmax(tuned_curve))
    res = {"tag": args.tag, "model": os.path.basename(args.model), "n": len(recs), "n_hidden": n_hidden,
           "frozen_lens_curve": frozen_curve, "tuned_lens_curve": tuned_curve,
           "frozen_peak": [fp, frozen_curve[fp]], "frozen_last": frozen_curve[-1],
           "tuned_peak": [tp, tuned_curve[tp]], "tuned_last": tuned_curve[-1]}
    # rotation OUT if tuned also peaks mid (>=0.75) and last collapses; cancellation confirmed by last~chance
    res["rotation_ruled_out"] = bool(tuned_curve[tp] >= 0.75 and tp < n_hidden - 2)
    res["cancellation_confirmed_output"] = bool(frozen_curve[-1] <= 0.65 and tuned_curve[-1] <= 0.70)
    print(f"\nfrozen lens: peak {frozen_curve[fp]:.3f}@L{fp}({fp/(n_hidden-1):.0%}) -> last {frozen_curve[-1]:.3f}")
    print(f"tuned  lens: peak {tuned_curve[tp]:.3f}@L{tp}({tp/(n_hidden-1):.0%}) -> last {tuned_curve[-1]:.3f}")
    print(f"rotation ruled out (tuned mid-peak real): {res['rotation_ruled_out']}")
    print(f"cancellation confirmed (both readouts ~chance at output): {res['cancellation_confirmed_output']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
