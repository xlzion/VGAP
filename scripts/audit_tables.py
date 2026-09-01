"""E0 numerical audit: every number in the paper's tables vs the json that produced it.
PASS if round(json,3)==tex value (values are printed to 3 decimals in the tex). FAIL otherwise.
"""
import json, os
V = "outputs/vgap_mech/"
def L(f):
    p = V + f
    if not os.path.exists(p): return None
    return json.load(open(p))
def g(d, *ks):
    for k in ks:
        if d is None: return None
        d = d.get(k) if isinstance(d, dict) else (d[k] if isinstance(d, list) else None)
    return d
def pt(x):  # first element if [pt,lo,hi], else scalar
    return x[0] if isinstance(x, list) else x

PASS=[]; FAIL=[]
def chk(tag, tex, jsonval):
    if jsonval is None: FAIL.append((tag, tex, "MISSING")); return
    ok = round(float(jsonval), 3) == round(float(tex), 3)
    (PASS if ok else FAIL).append((tag, tex, round(float(jsonval),4)))

H = ["qwen7b","llama8b","gemma4b"]
# ---------- tab:gap ----------
gapR = {"qwen7b":(.551,.539,.974),"llama8b":(.561,.535,.976),"gemma4b":(.538,.526,.962)}
for h,(gen,fl,pr) in gapR.items():
    d=L(f"S1_logitlens_rjudge_{h}.json"); e=L(f"E1_verbal_rjudge_{h}.json")
    chk(f"gap R-Judge {h} probe", pr, g(d,"probe_last"))
    chk(f"gap R-Judge {h} final-logit", fl, pt(g(d,"step0_final_logit_auroc")))
    chk(f"gap R-Judge {h} generated", gen, pt(g(e,"verbal","injection","verbal_auroc")))
gapA = {"qwen7b":(.564,.803),"llama8b":(.627,.783),"gemma4b":(.690,.790)}
for h,(fl,pr) in gapA.items():
    d=L(f"S1_logitlens_a3s_{h}.json")
    chk(f"gap A3S {h} probe", pr, g(d,"probe_last"))
    chk(f"gap A3S {h} final-logit", fl, pt(g(d,"step0_final_logit_auroc")))
urlR = {"qwen7b":(.968,.974),"llama8b":(.923,.977),"gemma4b":(.920,.914)}
for h,(gen,fl) in urlR.items():
    d=L(f"S1_URLcontrol_{h}.json"); e=L(f"E1_verbal_url_{h}.json")
    chk(f"gap URL {h} final-logit", fl, pt(g(d,"step0_final_logit_auroc")))
    chk(f"gap URL {h} generated", gen, pt(g(e,"verbal","url","verbal_auroc")))

# ---------- tab:battery ----------
def rd(c,h):
    if c=="injection": d=L(f"S1_logitlens_rjudge_{h}.json")
    elif c=="harmful": d=L(f"S1_concept_harmful_{h}.json")
    elif c=="url": d=L(f"S1_URLcontrol_{h}.json")
    else: d=L(f"S1_concept_{c}_{h}.json")
    return pt(g(d,"step0_final_logit_auroc"))
def vb(c,h):
    if c=="url": return pt(g(L(f"E1_verbal_url_{h}.json"),"verbal","url","verbal_auroc"))
    if c=="harmful":
        d=L("E1_verbal_audit_qwen7b.json") if h=="qwen7b" else L(f"E1_verbal_harmful_{h}.json")
        return pt(g(d,"verbal","harmful","verbal_auroc"))
    return pt(g(L(f"E1_verbal_rjudge_{h}.json"),"verbal",c,"verbal_auroc"))
BAT={"override":((.998,1.000,.961),(.993,.996,.890)),"email":((.993,.993,.971),(.984,.971,.940)),
 "url":((.974,.977,.914),(.968,.923,.920)),"code":((.557,.667,.652),(.530,.605,.654)),
 "injection":((.539,.535,.526),(.551,.561,.538)),"harmful":((.488,.504,.519),(.496,.497,.508))}
for c,(rr,vv) in BAT.items():
    for h,r,v in zip(H,rr,vv):
        chk(f"battery {c} readout {h}", r, rd(c,h)); chk(f"battery {c} verbal {h}", v, vb(c,h))
of=L("S1_injection_overridefree_qwen7b.json")
chk("battery override-free probe", .945, g(of,"probe_last"))
chk("battery override-free final-logit", .529, pt(g(of,"step0_final_logit_auroc")))

# ---------- tab:origin ----------
ORI={"email":(.624,.993),"override":(.577,.998),"url":(.514,.974),"injection":(.628,.539)}
for c,(base,inst) in ORI.items():
    if c=="url": bd=L("S1_URLcontrol_qwenBASE.json"); idv=rd("url","qwen7b")
    elif c=="injection": bd=L("S1_logitlens_rjudge_qwenBASE.json"); idv=rd("injection","qwen7b")
    else: bd=L(f"S1_concept_{c}_qwenBASE.json"); idv=rd(c,"qwen7b")
    chk(f"origin {c} base", base, pt(g(bd,"step0_final_logit_auroc")))
    chk(f"origin {c} instruct", inst, idv)
qb=L("S1_logitlens_rjudge_qwenBASE.json"); lb=L("M1_S1_rjudge_llama8b_base.json")
chk("origin Qwen-base final-logit", .628, pt(g(qb,"step0_final_logit_auroc")))
chk("origin Qwen-base probe", .969, g(qb,"probe_last"))
chk("origin Llama-base final-logit", .606, pt(g(lb,"step0_final_logit_auroc")))
chk("origin Llama-base probe", .962, g(lb,"probe_last"))

# ---------- tab:repair ----------
REP={"qwen7b":(.551,.432,.475,.974),"llama8b":(.561,.474,.629,.976),"gemma4b":(.538,.453,.628,.962)}
for h,(z,c,fw,pr) in REP.items():
    m=L(f"M2_intervention_rjudge_{h}.json")
    chk(f"repair {h} zeroshot", z, pt(g(m,"interventions","zeroshot","verbal_auroc")))
    chk(f"repair {h} cot", c, pt(g(m,"interventions","cot","verbal_auroc")))
    chk(f"repair {h} fewshot", fw, pt(g(m,"interventions","fewshot","verbal_auroc")))
    chk(f"repair {h} probe-ref", pr, g(L(f'S1_logitlens_rjudge_{h}.json'),"probe_last"))

# ---------- tab:dla ----------
DLA={"qwen7b":(.674,.482,.512,.013),"llama8b":(.624,.575,.514,.006),"gemma4b":(.616,.513,.524,.030)}
for h,(po,pn,nl,er) in DLA.items():
    d=L(f"G0_dla_{h}.json")
    chk(f"dla {h} pos-only", po, g(d,"pos_sum_auroc_OOF"))
    chk(f"dla {h} pos+neg", pn, g(d,"disc_only_auroc_OOF"))
    chk(f"dla {h} null-max", nl, g(d,"null_pos_only_max"))
    chk(f"dla {h} err", er, g(d,"dla_additivity_err"))

# ---------- tab:lens (R2 tuned) ----------
LENS={"qwen7b":(.944,.552,.774,.536),"llama8b":(.829,.536,.618,.532),"gemma4b":(.714,.529,.537,.527)}
for h,(fp,flz,tp,tl) in LENS.items():
    r=L(f"R2_tunedlens_{h}.json")
    s=L(f"S1_logitlens_rjudge_{h}.json")
    cb=[p[0] for p in s["curveB_lens"]]  # frozen peak from Fig 2a source (caption)
    chk(f"lens {h} frozen-peak", fp, max(cb))
    chk(f"lens {h} frozen-last", flz, g(r,"frozen_last"))
    chk(f"lens {h} tuned-peak", tp, g(r,"tuned_peak")[1])   # [layer, auroc]
    chk(f"lens {h} tuned-last", tl, g(r,"tuned_last"))

# ---------- tab:swap (subset-based; each value from its E4 record) ----------
SW=[("instruct-all override",1.000,"E4_override_f0.5.json","readout_orig"),
    ("instruct-all attack",.573,"E4_injection_f0.5.json","readout_orig"),
    ("inst[:14]+base override",1.000,"E4_override_f0.5.json","readout_after_swap"),
    ("inst[:14]+base attack",.552,"E4_injection_f0.5.json","readout_after_swap"),
    ("inst[:7]+base override",.899,"E4_override_f0.25.json","readout_after_swap"),
    ("base[:7]+inst override",.951,"E4_4thcell_override.json","readout_after_swap"),
    ("base[:7]+inst attack",.550,"E4_4thcell_injection.json","readout_after_swap"),
    ("base-all override",.492,"E4_4thcell_override.json","readout_orig"),
    ("base-all attack",.708,"E4_4thcell_injection.json","readout_orig")]
for tag,tex,f,fld in SW:
    chk(f"swap {tag}", tex, g(L(f),fld))

# ---------- report ----------
print(f"PASS {len(PASS)} / FAIL {len(FAIL)}\n")
if FAIL:
    print("=== FAIL / MISMATCH ===")
    for t,tex,jv in FAIL: print(f"  [{t}]  tex={tex}  json={jv}")
else:
    print("all clean")

# ---------- tab:swap: dump E4 records (subset-based; manual map) ----------
print("\n=== tab:swap raw E4 records (verify by hand) ===")
for f in ["E4_override_f0.5.json","E4_override_f0.25.json","E4_injection_f0.5.json",
          "E4_4thcell_override.json","E4_4thcell_injection.json","E4_TINY.json"]:
    d=L(f)
    if d: print(f"  {f}: concept={d.get('concept')} swap_from={d.get('swap_from_layer')} "
                f"orig={d.get('readout_orig')} after={d.get('readout_after_swap')} n={d.get('n')}")
