"""A1 figure: per-layer probe vs the model's own output readout (frozen logit-lens),
6 panels = {R-Judge, A3S} x {Qwen2.5-7B, llama-3-8b, gemma-3-4b}.

Probe (external readout) reads the injection at every layer; the model's own output readout
(frozen lens) peaks mid-network then collapses to chance. Authoritative last-layer output = Step0
(the true logits) is marked separately, since the frozen lens double-norms the last point.
Renders locally (matplotlib venv). Reads outputs/vgap_mech/S1_logitlens_{rjudge,a3s}_{host}.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = "outputs/vgap_mech/"
HOSTS = [("qwen7b", "Qwen2.5-7B"), ("llama8b", "Llama-3-8B"), ("gemma4b", "Gemma-3-4B")]
DSETS = [("rjudge", "R-Judge (n=571)"), ("a3s", "A3S (n=1150)")]
C_PROBE, C_LENS = "#1f77b4", "#d62728"

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True, sharey=True)
for i, (ds, dsname) in enumerate(DSETS):
    for j, (hk, hn) in enumerate(HOSTS):
        ax = axes[i][j]
        d = json.load(open(f"{V}S1_logitlens_{ds}_{hk}.json"))
        A = np.array([x[0] for x in d["curveA_probe"]])
        B = np.array([x[0] for x in d["curveB_lens"]])
        L = len(A) - 1
        xs = np.arange(L + 1) / L
        step0 = d["step0_final_logit_auroc"][0]
        ax.axhline(0.5, color="grey", lw=0.8, ls=":", zorder=1)
        ax.plot(xs, A, color=C_PROBE, lw=2.0, label="probe (activation)", zorder=3)
        ax.plot(xs, B, color=C_LENS, lw=1.8, ls="--", label="output readout (lens)", zorder=3)
        ax.plot([1.0], [step0], marker="o", ms=6, mfc="white", mec=C_LENS, mew=1.6,
                zorder=4, label="output readout (Step0, true logits)")
        pk = int(np.argmax(B))
        ax.annotate(f"{B[pk]:.2f}", (pk / L, B[pk]), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color=C_LENS)
        ax.set_ylim(0.42, 1.02)
        ax.set_title(hn, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{dsname}\nAUROC", fontsize=10)
        if i == 1:
            ax.set_xlabel("normalized layer depth", fontsize=10)
        ax.grid(True, alpha=0.25)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.045, 1, 1])
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/A1_layer_curves.{ext}", dpi=170, bbox_inches="tight")
print("saved figures/A1_layer_curves.{png,pdf}")
# also dump the plotted numbers for provenance
prov = {f"{ds}_{hk}": {"probe": [round(x[0], 4) for x in json.load(open(f'{V}S1_logitlens_{ds}_{hk}.json'))["curveA_probe"]],
                        "lens":  [round(x[0], 4) for x in json.load(open(f'{V}S1_logitlens_{ds}_{hk}.json'))["curveB_lens"]],
                        "step0": json.load(open(f'{V}S1_logitlens_{ds}_{hk}.json'))["step0_final_logit_auroc"][0]}
        for ds, _ in DSETS for hk, _ in HOSTS}
json.dump(prov, open("figures/A1_layer_curves_data.json", "w"), indent=1)
print("saved figures/A1_layer_curves_data.json")
