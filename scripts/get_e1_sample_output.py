"""Persist ONE judge-on-attack sample so a figure number has a provenance file.

Regenerates the safety-judge's actual output for a single R-Judge record (greedy, deterministic)
and prints a JSON record to STDOUT (all logs go to STDERR, so `> file.json` captures clean JSON).
Used for the Figure 1 teaser: the model outputs a bare score, not a sentence.

  CUDA_VISIBLE_DEVICES=0 python scripts/get_e1_sample_output.py --uid rjudge-Application-dh_app-1027 \
      > outputs/vgap_mech/e1_sample_app1027.json
"""
import argparse, json, os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nscc_pilot.adapters import load_dataset
from vgap_common import JUDGE_PROMPT, middle_out, parse_risk


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="data/rjudge_balanced.jsonl")
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = load_dataset("jsonl", {"jsonl_path": args.dataset})
    hits = [r for r in recs if r.uid == args.uid]
    if not hits:
        log(f"uid {args.uid} not found in {args.dataset}"); sys.exit(1)
    x = hits[0]
    log(f"found {args.uid}: label={x.label}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()

    ids = tok.encode(x.flat_text(), add_special_tokens=False)
    ctx = tok.decode(middle_out(ids, args.max_ctx), skip_special_tokens=True)
    msg = [{"role": "user", "content": JUDGE_PROMPT.format(ctx=ctx)}]
    prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    raw = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    score01 = parse_risk(raw)
    log(f"raw_output={raw!r}  score={int(round(score01 * 100))}/100")

    rec = {
        "uid": x.uid, "dataset": x.dataset, "label": int(x.label),
        "attack_type": (x.meta or {}).get("attack_type"), "category": (x.meta or {}).get("category"),
        "model": os.path.basename(args.model), "judge_prompt": "vgap_common.JUDGE_PROMPT",
        "decoding": "greedy (do_sample=False)", "max_new_tokens": args.max_new_tokens,
        "raw_output": raw, "parsed_score_0_1": score01, "score_0_100": int(round(score01 * 100)),
        "context_flat_text": x.flat_text(),
        "generated_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/get_e1_sample_output.py",
    }
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
