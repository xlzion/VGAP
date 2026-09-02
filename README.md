# MODELS KNOW BUT DON’T SAY: WHY LLM SAFETY JUDGES CANNOT VERBALIZE THE ATTACKS THEY DETECT

Code and experiment artifacts for studying the *know–say gap*: an LLM used as a
safety judge carries a near-perfect internal signal about whether a trajectory
contains a prompt injection (linear probe on activations), while its own
verbalized score stays close to chance. This repository contains the scripts
that locate the gap layer by layer, test whether prompting can close it,
attribute it to output-path cancellation, and trace its origin in
base-vs-instruct weights — together with the JSON outputs and run logs behind
the paper's tables and figures.

## Layout

```
scripts/    experiment entry points (one file per experiment)
src/        vendored evaluation harness + vgap_common.py (judge prompt,
            parsing, probe utilities, concept definitions)
data/       balanced evaluation sets: rjudge_balanced.jsonl, a3s_balanced.jsonl
outputs/vgap_mech/   all result JSONs and run logs
figures/    paper figures (PDF) and the layer-curve figure's source data
```

## Scripts

| Script | What it does |
|---|---|
| `run_vgap_confirm.py` | Confirmation experiment: verbal score vs. activation probe on the injection channel, with an objective URL-detection control channel. |
| `run_s1_logitlens.py` | Layer-wise probe and logit-lens curves; `--concept` selects the concept battery target (injection, url, email, code, override, harmful). |
| `run_verbal_battery.py` | Generation-level verbal scores per concept (E1). |
| `run_m2_interventions.py` | Prompting interventions (zero-shot / CoT / few-shot) — does asking better close the gap? |
| `run_b0_erasure.py` | Additive residual-stream decomposition; per-component erasure attribution. |
| `run_b1_ablation.py` | Mean-ablation of top erasing components. |
| `run_g0_dla.py` | Frozen-norm direct logit attribution with out-of-fold component selection and a label-permutation null. |
| `run_g1_suppression.py` | Peak-vs-final decoupling analysis. |
| `run_g2_patch.py` | Surgical removal of negative-discriminative components' projection on the readout direction. |
| `run_r2_tunedlens.py` | Frozen-lens and tuned-lens mid-layer readout. |
| `run_e4_layerswap.py` | Base/instruct layer-weight interchange (origin attribution). |
| `get_e1_sample_output.py` | Dump a single sample's generated judgment. |
| `make_A1_layer_curves.py` | Figure: layer curves (probe / lens / final-logit readout). |
| `gen_A2_ci_table.py` | Table: concept battery with bootstrap CIs. |
| `audit_tables.py` | Re-derives every table cell from the JSONs in `outputs/` and checks them. |

Typical invocation (see each script's docstring for its exact flags):

```
python scripts/run_s1_logitlens.py \
    --model Qwen/Qwen2.5-7B-Instruct --tag rjudge_qwen7b \
    --data data/rjudge_balanced.jsonl --out outputs/vgap_mech
```

Model arguments accept a Hugging Face model id or a local path. Experiments in
the paper use Qwen2.5-7B-Instruct (and its base variant), Llama-3-8B-Instruct,
and Gemma-3-4b-it.

## Environment

Python 3.10, torch 2.11.0, transformers 5.8.1, scikit-learn 1.7.2, numpy 2.2.6
(`requirements.txt`); matplotlib for the figure scripts. Single GPU (tested on
32 GB); smaller hosts fit in less.

## Outputs

`outputs/vgap_mech/` ships the JSON results and logs produced by the runs
reported in the paper, so tables and figures can be regenerated (or audited via
`audit_tables.py`) without re-running the models.
