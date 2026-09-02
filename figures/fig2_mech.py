# Fig 2 — the know–say gap, quantitatively.
# (a) per-layer decodability of the injection judgment (Qwen2.5-7B, R-Judge)
# (b) concept battery: verbal AUROC, 6 concepts x 3 hosts
# All numbers verified against outputs/vgap_mech/ (结果表汇总_20260901.md 表1/表2).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

C_CONTENT = "#0072B2"
C_META    = "#D55E00"
C_MID     = "#6B7280"
INK       = "#1F2328"
MUT       = "#6B7280"
GRID      = "#D7D9DC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.color": INK,
    "axes.edgecolor": "#B7BABE",
    "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ---------------- real data: S1_logitlens_rjudge_qwen7b.json ----------------
LENS    = [0.5, 0.2893, 0.285, 0.181, 0.5417, 0.6719, 0.8546, 0.818, 0.8861,
           0.9443, 0.9313, 0.9048, 0.9063, 0.8722, 0.8188, 0.7486, 0.8183,
           0.8479, 0.8113, 0.5382, 0.4592, 0.5154, 0.5117, 0.5289, 0.5282,
           0.5285, 0.5466, 0.518, 0.5522]
LENS_LO = [0.5, 0.2448, 0.241, 0.1455, 0.4956, 0.6253, 0.822, 0.7823, 0.8565,
           0.925, 0.9115, 0.8813, 0.8801, 0.8419, 0.7837, 0.7078, 0.7804,
           0.8135, 0.7713, 0.4885, 0.4157, 0.4691, 0.4641, 0.4811, 0.4808,
           0.4802, 0.4982, 0.4703, 0.5049]
LENS_HI = [0.5, 0.3341, 0.3259, 0.2168, 0.591, 0.7141, 0.8859, 0.8542, 0.9122,
           0.9623, 0.9518, 0.9255, 0.9301, 0.9005, 0.8515, 0.7872, 0.8526,
           0.878, 0.848, 0.5856, 0.5088, 0.5646, 0.5591, 0.5766, 0.575,
           0.5747, 0.5922, 0.5657, 0.5997]
PROBE    = [0.5, 0.9652, 0.9662, 0.9649, 0.9639, 0.9647, 0.9668, 0.9656,
            0.9664, 0.9651, 0.9669, 0.9667, 0.9651, 0.9662, 0.966, 0.9647,
            0.9705, 0.9697, 0.9711, 0.9732, 0.9739, 0.9713, 0.9708, 0.9704,
            0.9717, 0.9707, 0.9683, 0.9716, 0.9739]
PROBE_LO = [0.5, 0.9511, 0.9518, 0.9501, 0.9494, 0.9493, 0.9516, 0.9498,
            0.9505, 0.9494, 0.953, 0.9532, 0.9498, 0.9535, 0.9528, 0.9506,
            0.9582, 0.9575, 0.9593, 0.9607, 0.9615, 0.9574, 0.9568, 0.957,
            0.9582, 0.9573, 0.9533, 0.959, 0.9612]
PROBE_HI = [0.5, 0.9778, 0.9784, 0.9778, 0.9764, 0.9774, 0.9795, 0.9788,
            0.9797, 0.9785, 0.9797, 0.9789, 0.9785, 0.9785, 0.9788, 0.9778,
            0.9817, 0.9812, 0.9822, 0.983, 0.984, 0.9826, 0.9824, 0.9815,
            0.9826, 0.9819, 0.98, 0.9818, 0.9831]
PROBE_LAST = 0.9739
STEP0      = 0.5388     # true final-logit ("output") readout, Step0

# ---------------- real data: E1_verbal_* (verbal AUROC [pt, lo, hi]) --------
VERBAL = {
    "override": dict(Q=[0.9929, 0.9851, 0.9983], L=[0.9958, 0.9903, 1.0],
                     G=[0.8900, 0.8600, 0.9189], kind="content"),
    "email":    dict(Q=[0.9836, 0.9741, 0.9919], L=[0.9705, 0.9549, 0.9846],
                     G=[0.9402, 0.9175, 0.9595], kind="content"),
    "url":      dict(Q=[0.9676, 0.9501, 0.9820], L=[0.9227, 0.8962, 0.9462],
                     G=[0.9204, 0.8838, 0.9526], kind="content"),
    "code":     dict(Q=[0.5303, 0.4917, 0.5668], L=[0.6053, 0.5622, 0.6452],
                     G=[0.6537, 0.6232, 0.6852], kind="mid"),
    "injection":dict(Q=[0.5509, 0.5135, 0.5866], L=[0.5609, 0.5153, 0.6045],
                     G=[0.5378, 0.4915, 0.5836], kind="meta"),
    "harmful":  dict(Q=[0.4957, 0.4488, 0.5398], L=[0.4967, 0.4528, 0.5401],
                     G=[0.5083, 0.4622, 0.5559], kind="meta"),
}
PROBE_BAND = (0.9618, 0.9759)   # probe_last across the 3 hosts
HOST_MARK = {"Q": "o", "L": "s", "G": "^"}
HOST_NAME = {"Q": "Qwen2.5-7B", "L": "Llama-3-8B", "G": "Gemma-3-4B"}
KIND_COLOR = {"content": C_CONTENT, "mid": C_MID, "meta": C_META}

# ================= figure =================
fig = plt.figure(figsize=(7.2, 2.85))
axA = fig.add_axes([0.070, 0.175, 0.345, 0.655])
axB = fig.add_axes([0.615, 0.175, 0.375, 0.655])

# ------------------------------------------------ (a) per-layer decodability
L = np.arange(29)
axA.set_title("(a)  “attack?” decodability across layers", fontsize=7.6,
              fontweight="bold", color=INK, loc="left", pad=8)
axA.axhline(0.5, color=MUT, lw=0.7, ls=(0, (3, 2)), zorder=1)
axA.fill_between(L, PROBE_LO, PROBE_HI, color=INK, alpha=0.13, lw=0, zorder=2)
axA.plot(L, PROBE, ls=(0, (3, 2)), lw=1.1, color=INK, zorder=3)
axA.fill_between(L, LENS_LO, LENS_HI, color=C_META, alpha=0.15, lw=0, zorder=2)
axA.plot(L, LENS, lw=1.4, color=C_META, zorder=4)
axA.plot(28, PROBE_LAST, "o", ms=3.6, mfc=INK, mec="white", mew=0.5, zorder=5)
axA.plot(28, STEP0, "D", ms=3.4, mfc=C_META, mec="white", mew=0.5, zorder=5)
axA.text(28.9, 0.965, "probe\n0.97", fontsize=5.3, color=INK, ha="left",
         va="center", linespacing=1.15)
axA.text(28.9, 0.545, "output\n0.54", fontsize=5.3, color=C_META, ha="left",
         va="center", linespacing=1.15)

axA.text(14.0, 1.005, "linear probe (hidden state)", fontsize=5.6, color=INK,
         ha="center", va="center")
axA.text(19.8, 0.665, "logit lens (output readout)", fontsize=5.6,
         color=C_META, ha="left", va="center")
axA.text(9.8, 0.70, "briefly readable", fontsize=5.3, color=C_META,
         ha="center", va="center", style="italic")
axA.text(23.0, 0.335, "then discarded", fontsize=5.3, color=C_META,
         ha="center", va="center", style="italic")
axA.text(0.7, 0.455, "chance", fontsize=5.3, color=MUT, ha="left", va="top")

axA.set_xlim(-0.6, 33.6)
axA.set_ylim(0.10, 1.04)
axA.set_xticks([0, 7, 14, 21, 28])
axA.set_yticks([0.25, 0.50, 0.75, 1.00])
axA.set_xticklabels(["0", "7", "14", "21", "28"], fontsize=6.2)
axA.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.2)
axA.set_xlabel("layer · Qwen2.5-7B · R-Judge (n = 571)", fontsize=6.6,
               labelpad=2)
axA.set_ylabel("AUROC", fontsize=6.6, labelpad=2)
axA.tick_params(length=2, width=0.6)
for s in ("top", "right"): axA.spines[s].set_visible(False)
axA.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axA.set_axisbelow(True)

# ------------------------------------------------ (b) concept battery
order = ["override", "email", "url", "code", "injection", "harmful"]
ylab = {"override": "override phrase", "email": "email",
        "url": "URL", "code": "code",
        "injection": "attack?", "harmful": "harmful?"}
ypos = {c: len(order) - 1 - i for i, c in enumerate(order)}
JIT = {"Q": 0.17, "L": 0.0, "G": -0.17}

axB.set_title("(b)  concept battery (verbal readout)", fontsize=7.6,
              fontweight="bold", color=INK, loc="left", pad=8)
axB.axvline(PROBE_LAST, color="#9AA0A6", lw=1.0, zorder=1)
axB.text(PROBE_LAST, 5.66, "probe 0.97", fontsize=5.8, color="#4B5157",
         ha="center", va="bottom")
axB.axvline(0.5, color=MUT, lw=0.8, ls=(0, (3, 2)), zorder=1)
axB.text(0.5, 5.66, "chance", fontsize=5.8, color=MUT, ha="center",
         va="bottom")

for c in order:
    d = VERBAL[c]
    col = KIND_COLOR[d["kind"]]
    axB.plot(d["Q"][0], ypos[c], "o", ms=6.0, mfc=col, mec="white", mew=0.8,
             zorder=4)

axB.set_xlim(0.4, 1.005)
axB.set_ylim(-0.62, 5.62)
axB.set_yticks([ypos[c] for c in order])
axB.set_yticklabels([ylab[c] for c in order], fontsize=6.3)
for tl, c in zip(axB.get_yticklabels(), order):
    tl.set_color(KIND_COLOR[VERBAL[c]["kind"]])
axB.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                    fontsize=6.2)
axB.set_xlabel("verbal AUROC · Qwen2.5-7B · R-Judge", fontsize=6.6,
               labelpad=2)
axB.tick_params(length=2, width=0.6)
for s in ("top", "right"): axB.spines[s].set_visible(False)
axB.grid(axis="x", color=GRID, lw=0.5, zorder=0)
axB.set_axisbelow(True)

axB.text(0.415, ypos["email"] + 0.5, "content", fontsize=6.4, color=C_CONTENT,
         ha="left", va="center", fontweight="bold")
axB.text(0.660, ypos["injection"] - 0.5, "judgment", fontsize=6.4,
         color=C_META, ha="left", va="center", fontweight="bold")

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"figures/fig2_mech.{ext}", dpi=300, facecolor="white")
print("done")
