"""E1: verbal-generation concept battery -- the 'say' half of know-vs-say.

The frozen-lens battery is a READOUT-level claim (what the model WOULD output). E1 makes it a
GENERATION claim: actually greedily generate the 0-100 judgment for each concept and score it.
Crown jewel: the model can SAY 'there is an override phrase' (~0.9) but cannot SAY 'this is an
attack' (~0.55) -- same context, same format, only the question differs. Content is verbalizable;
the adversarial/evaluative judgment is not.

  CUDA_VISIBLE_DEVICES=0 python scripts/run_verbal_battery.py --model Qwen/Qwen2.5-7B-Instruct \
      --concepts injection,override,email,code,url --tag rjudge_qwen7b \
      --out outputs/vgap_mech/E1_verbal_rjudge_qwen7b.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from nscc_pilot.adapters import load_dataset
from nscc_pilot.metrics import roc_auc, bootstrap_ci
import re as _re
from vgap_common import JUDGE_PROMPT, CONTROL_PROMPT, CONCEPTS, control_label, middle_out, parse_risk
from vgap_common import HARMFUL_PROMPT


def _has_number(text):
    t = text.split("</think>")[-1] if "</think>" in text else text
    return bool(_re.search(r"\d", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--tag", default="rjudge_qwen7b")
    ap.add_argument("--concepts", default="injection,override,email,code,url")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="outputs/vgap_mech/E1_verbal_rjudge_qwen7b.json")
    args = ap.parse_args()
    concepts = args.concepts.split(",")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    if args.n:
        recs = recs[: args.n]
    texts = [r.flat_text() for r in recs]
    reclab = np.array([int(r.label) for r in recs])
    labels = {"injection": reclab, "harmful": reclab, "url": np.array([control_label(t) for t in texts])}
    for c in concepts:
        if c in CONCEPTS and CONCEPTS[c][2] is not None:
            labels[c] = np.array([CONCEPTS[c][2](t) for t in texts])
    prompts = {"injection": JUDGE_PROMPT, "harmful": HARMFUL_PROMPT, "url": CONTROL_PROMPT}
    for c in concepts:
        if c in CONCEPTS and c not in ("injection", "harmful") and CONCEPTS[c][1] is not None:
            prompts[c] = CONCEPTS[c][1]
    print(f"{args.tag}: {len(recs)} records | concepts={concepts}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map=os.environ.get("NSCC_DEVICE_MAP", "cuda")).eval()

    def build_enc(prompt, ctx):
        ids = tok.encode(ctx, add_special_tokens=False)
        ctx = tok.decode(middle_out(ids, args.max_ctx), skip_special_tokens=True)
        msg = [{"role": "user", "content": prompt.format(ctx=ctx)}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        if tok.chat_template:
            try:
                return tok.apply_chat_template(msg, enable_thinking=False, **kw)
            except TypeError:
                return tok.apply_chat_template(msg, **kw)
        return tok(prompt.format(ctx=ctx), return_tensors="pt")

    res = {"tag": args.tag, "model": os.path.basename(args.model), "n": len(recs), "verbal": {}}
    with torch.no_grad():
        for c in concepts:
            if c not in prompts or c not in labels:
                print(f"  skip {c} (no prompt/label)"); continue
            y = labels[c]; verbal = []; parsed = []
            for t in texts:
                enc = build_enc(prompts[c], t).to(model.device)
                gen = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
                out = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                verbal.append(parse_risk(out)); parsed.append(_has_number(out))
            verbal = np.array(verbal); parsed = np.array(parsed)
            a, lo, hi = bootstrap_ci(roc_auc, verbal, y, n_boot=2000)
            # audit the bad-metric risk: real parse-failure rate per class + AUROC on parsed-only
            fail_pos = float(np.mean(~parsed[y == 1])); fail_neg = float(np.mean(~parsed[y == 0]))
            po = parsed & True
            a_parsed = float(roc_auc(verbal[po], y[po])) if (po.sum() > 5 and len(np.unique(y[po])) == 2) else float("nan")
            res["verbal"][c] = {"type": CONCEPTS.get(c, ("meta",))[0], "verbal_auroc": [a, lo, hi],
                                "n_pos": int(y.sum()), "parse_fail_pos": fail_pos, "parse_fail_neg": fail_neg,
                                "auroc_parsed_only": a_parsed, "frac_no_number": float(np.mean(~parsed))}
            print(f"  [{c:10s}] AUROC={a:.3f} [{lo:.2f},{hi:.2f}] | parsed-only={a_parsed:.3f} | "
                  f"no-number pos/neg={fail_pos:.2f}/{fail_neg:.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
