"""Regenerate the concept-battery CI appendix table DIRECTLY from the result json
(discipline: numbers must trace to the file that produced them; do not trust stale tex).
Emits a colored table matching the master style (ContentBG/MidBG/JudgeBG, bold judgment).
Fails loudly if any source file/key is missing.
"""
import json, os, sys
V = "outputs/vgap_mech/"
H = ["qwen7b", "llama8b", "gemma4b"]

def L(f):
    p = V + f
    if not os.path.exists(p): sys.exit(f"MISSING: {p}")
    return json.load(open(p))

def readout_ci(c, h):
    if c == "injection": d = L(f"S1_logitlens_rjudge_{h}.json")
    elif c == "harmful": d = L(f"S1_concept_harmful_{h}.json")
    elif c == "url":     d = L(f"S1_URLcontrol_{h}.json")
    else:                d = L(f"S1_concept_{c}_{h}.json")
    return d["step0_final_logit_auroc"]

def verbal_ci(c, h):
    if c == "url":
        d = L(f"E1_verbal_url_{h}.json"); return d["verbal"]["url"]["verbal_auroc"]
    if c == "harmful":
        d = L("E1_verbal_audit_qwen7b.json") if h == "qwen7b" else L(f"E1_verbal_harmful_{h}.json")
        return d["verbal"]["harmful"]["verbal_auroc"]
    d = L(f"E1_verbal_rjudge_{h}.json"); return d["verbal"][c]["verbal_auroc"]

# (concept key, display label, kind, row-color)
ROWS = [
    ("override", "override phrase present?", "content", "ContentBG", False),
    ("email",    "email present?",           "content", "ContentBG", False),
    ("url",      "URL present?",             "content", "ContentBG", False),
    ("code",     "code present?",            "middle",  "MidBG",     False),
    ("injection","is this an attack?",       "judgment","JudgeBG",   True),
    ("harmful",  "is this harmful / unsafe?","judgment","JudgeBG",   True),
]

def cell(ci):
    pt, lo, hi = ci
    return f"{pt:.3f}\\,[{lo:.2f},{hi:.2f}]"

def trio(fn, c, bold):
    s = " / ".join(cell(fn(c, h)) for h in H)
    return f"\\textbf{{{s}}}" if bold else s

out = []
out.append(r"\begin{table}[h]")
out.append(r"\centering\small")
out.append(r"\caption{\textbf{Concept battery with 95\% bootstrap confidence intervals} "
           r"(R-Judge, $n{=}571$; Q\,/\,L\,/\,G $=$ Qwen2.5-7B / Llama-3-8B / Gemma-3-4B). "
           r"\emph{readout} $=$ final-logit score, \emph{verbal} $=$ parsed greedy answer. "
           r"Row tint marks concept kind (\colorbox{ContentBG}{content} / "
           r"\colorbox{MidBG}{middle} / \colorbox{JudgeBG}{judgment}); judgment rows bold. "
           r"Every number is read from the per-concept result file.}")
out.append(r"\label{tab:battery-ci}")
out.append(r"\begin{tabularx}{\textwidth}{l l X}")
out.append(r"\hline")
out.append(r"concept (question) & readout/verbal & AUROC (95\% CI) \quad Q / L / G \\ \hline")
for c, lab, kind, col, bold in ROWS:
    lab_c = f"\\textbf{{{lab}}}" if bold else lab
    out.append(f"\\rowcolor{{{col}}} {lab_c} & {'\\textbf{readout}' if bold else 'readout'} & {trio(readout_ci, c, bold)} \\\\")
    out.append(f"\\rowcolor{{{col}}}  & {'\\textbf{verbal}' if bold else 'verbal'} & {trio(verbal_ci, c, bold)} \\\\ \\hline")
out.append(r"\end{tabularx}")
out.append(r"\end{table}")

dst = "paper_iclr2027/A2_concept_battery_CI.tex"
open(dst, "w").write("\n".join(out) + "\n")
print("wrote", dst)
print("\n".join(out))
