"""E4: causal test of §5 -- is the CONTENT readout channel located in the instruction-tuned
late-layer weights?

base and instruct (Qwen2.5-7B) share init and differ only by instruction tuning, so their weights
are close and swapping late layers stays coherent. We swap layers[k:] between the two and read the
frozen-lens output readout for a content concept (override) and the judgment (injection):
  - instruct with BASE late layers: does the content readout DROP (channel removed)?
  - base with INSTRUCT late layers: does the content readout APPEAR (channel installed)?
If content readout tracks the late-layer weights while the injection readout stays at chance in all
four cells, the late-layer weights ARE the granted content channel, and instruction tuning installs
no channel for the adversarial judgment.

Sequential loading (never two 7B on GPU): load donor, lift layers[k:] state_dict to CPU, free it,
load recipient, overwrite recipient.layers[k:], run.

  CUDA_VISIBLE_DEVICES=0 python scripts/run_e4_layerswap.py \
      --recipient Qwen/Qwen2.5-7B-Instruct --donor Qwen/Qwen2.5-7B \
      --swap_frac 0.75 --concept override --out outputs/vgap_mech/E4_swap_instr_with_base_override.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
from vgap_common import JUDGE_PROMPT, CONTROL_PROMPT, CONCEPTS, control_label, middle_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--donor", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--concept", default="override")
    ap.add_argument("--swap_frac", type=float, default=0.75, help="swap layers[k:] with k=frac*L")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="outputs/vgap_mech/E4_swap.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    if args.n:
        recs = recs[: args.n]
    texts = [r.flat_text() for r in recs]
    if args.concept == "injection":
        y = np.array([int(r.label) for r in recs]); PROMPT = JUDGE_PROMPT
    elif args.concept == "url":
        y = np.array([control_label(t) for t in texts]); PROMPT = CONTROL_PROMPT
    else:
        y = np.array([CONCEPTS[args.concept][2](t) for t in texts]); PROMPT = CONCEPTS[args.concept][1]

    tok = AutoTokenizer.from_pretrained(args.recipient, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def core_of(m):
        c = m.model if hasattr(m, "model") else m
        return getattr(c, "language_model", None) or c

    # 1) load donor, lift late-layer weights to CPU, free donor
    donor = AutoModelForCausalLM.from_pretrained(args.donor, trust_remote_code=True,
                                                 dtype=torch.bfloat16, device_map="cpu")
    dl = core_of(donor).layers; L = len(dl); k = int(args.swap_frac * L)
    donor_state = {i: {n: p.detach().clone() for n, p in dl[i].state_dict().items()} for i in range(k, L)}
    del donor; torch.cuda.empty_cache()
    print(f"lifted donor layers[{k}:{L}] to CPU", flush=True)

    # 2) load recipient
    model = AutoModelForCausalLM.from_pretrained(args.recipient, trust_remote_code=True,
                                                 dtype=torch.bfloat16, device_map="cuda").eval()
    rl = core_of(model).layers
    lm_head = model.get_output_embeddings()
    fn = getattr(core_of(model), "norm", None) or getattr(core_of(model), "final_layernorm", None)
    softcap = getattr(model.config, "final_logit_softcapping", None)

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
        msg = [{"role": "user", "content": PROMPT.format(ctx=ctx)}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        if tok.chat_template:
            try:
                return tok.apply_chat_template(msg, enable_thinking=False, **kw)
            except TypeError:
                return tok.apply_chat_template(msg, **kw)
        return tok(PROMPT.format(ctx=ctx), return_tensors="pt")

    @torch.no_grad()
    def readout():
        s = []
        for t in texts:
            enc = build_enc(t).to(model.device)
            s.append(exp_number(model(**enc, use_cache=False).logits[0, -1]))
        s = np.array(s)
        return bootstrap_ci(roc_auc, s, y, n_boot=1000)   # (point, lo, hi)

    o, olo, ohi = readout()
    # 3) overwrite recipient late layers with donor's, re-read
    with torch.no_grad():
        for i in range(k, L):
            rl[i].load_state_dict({n: p.to(model.device) for n, p in donor_state[i].items()})
    sw, swlo, swhi = readout()
    orig, swapped = float(o), float(sw)

    res = {"recipient": os.path.basename(args.recipient), "donor": os.path.basename(args.donor),
           "concept": args.concept, "swap_from_layer": k, "n_layers": L, "n": len(recs),
           "n_pos": int(y.sum()), "readout_orig": orig, "readout_orig_ci": [float(olo), float(ohi)],
           "readout_after_swap": swapped, "readout_after_swap_ci": [float(swlo), float(swhi)],
           "delta": swapped - orig}
    print(f"\n[{args.concept}] recipient({os.path.basename(args.recipient)}) readout={orig:.3f} "
          f"-> with donor({os.path.basename(args.donor)}) layers[{k}:]={swapped:.3f} (delta {swapped-orig:+.3f})", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
