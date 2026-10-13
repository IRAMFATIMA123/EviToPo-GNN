# EviTopo-GNN

**Evidential Vessel-Topology Graph Learning with Uncertainty-Guided Contrastive Representation for Retinal Vessel Segmentation in Diabetic Retinopathy**

Iram Fatima, Ihtiram Raza Khan, Harleen Kaur — School of Engineering Sciences and Technology, Jamia Hamdard, New Delhi, India

This repository is a from-scratch, equation-faithful **PyTorch reference implementation** of the EviTopo-GNN architecture described in the paper, plus the data/training/evaluation pipeline needed to reproduce its protocol, and the **Table 1–5 results and figures** from the Results section, regenerated from the paper's own published numbers (see [Reproducing the paper's tables and figures](#reproducing-the-papers-tables-and-figures) for exactly what that means and doesn't mean).

> Code, weights, split indices, seeds and per-image predictions for the *original* study are intended to be released at `https://github.com/<author-account>/EviTopo-GNN` (a placeholder in the manuscript at the time of writing). This repository is an independent, from-scratch reimplementation of that architecture and protocol — it is not that original codebase, and does not ship the original trained weights.

---

## What EviTopo-GNN does

Retinal vessel segmentation (RVS) measures the vascular tree that downstream diabetic-retinopathy grading depends on. Pixel-wise models fragment thin capillaries, confuse lesion edges for vessels, and give no signal of where they're wrong. EviTopo-GNN couples three mechanisms that prior work studied in isolation:

1. **Evidential uncertainty** — a Dirichlet evidential head yields a calibrated probability *and* a pixel-wise uncertainty from a single forward pass (no Monte Carlo sampling).
2. **Explicit vessel-topology graph reasoning** — a graph is rebuilt every forward pass on the predicted centreline, with orientation-gated candidate edges between dangling endpoints; an uncertainty-gated GNN moves confident evidence along the tree, and an evidential link head decides, edge by edge, which gaps to bridge.
3. **Topology-aware, uncertainty-guided contrastive learning** — positives are defined by geodesic connectivity on the ground-truth vessel graph, and anchors are drawn preferentially from uncertain pixels.

| Fig. 1 — the five-stage pipeline (Eq. 1) |
|---|
| ![pipeline](results/figures/fig1_pipeline.png) |

## Headline results (Table 1)

clDice up to **0.876** (FIVES), a **23–35%** reduction in the zero-dimensional Betti error, and a **2.3–2.9×** reduction in expected calibration error (ECE), with small and mostly non-significant pixel-F1 gains — the contribution is topological and reliability-related, not a new pixel-overlap backbone. See [`results/`](results/) for the full Table 1–5 reproduction.

![F1 and clDice by method and dataset](results/figures/fig2_f1_cldice_by_dataset.png)

---

## Repository layout

```
EviTopo-GNN/
├── evitopo_gnn/                 # the installable package
│   ├── models/                  # the network, split one module per Fig. 1 stage
│   │   ├── lora.py                  Eq. 2  — LoRA adapters for the frozen ViT
│   │   ├── hybrid_encoder.py        Sec 3.2 — ViT-CNN encoder + windowed cross-attn (Eq. 3)
│   │   ├── coarse_decoder.py        Sec 3.2-3.3 — top-down fusion to one decoded feature map
│   │   ├── evidential_head.py       Eq. 4  — Dirichlet evidential head (coarse + final)
│   │   ├── skeleton.py              soft skeleton (clDice-style) + hard skeleton/pruning
│   │   ├── graph_builder.py         Sec 3.4, Eq. 6 — vessel-topology graph + gap candidates
│   │   ├── gnn_layers.py            Sec 3.5, Eq. 7-9 — uncertainty-gated graph attention
│   │   ├── link_predictor.py        Sec 3.6, Eq. 10-11 — evidential gap-link head
│   │   ├── decoder.py               Sec 3.7, Eq. 12 — graph-to-pixel splat decoder
│   │   ├── contrastive_head.py      Sec 3.8 — projection head for the contrastive loss
│   │   └── evitopo_gnn.py           Eq. 1 — wires all five stages together
│   ├── losses/                  # the training objective, one file per term
│   │   ├── evidential_loss.py       Eq. 5  — Bayes-risk evidential loss + KL-ramp
│   │   ├── dice_cldice_loss.py      soft Dice + differentiable clDice
│   │   ├── link_loss.py             Sec 3.6 — gap-link supervision + candidate labelling
│   │   ├── contrastive_loss.py      Eq. 13-14 — topology-aware uncertainty-guided contrastive
│   │   └── combined_objective.py    Eq. 15 + the 3-stage schedule (Sec 3.9)
│   ├── data/                    # dataset loaders, splits, synthetic-gap augmentation
│   ├── graph/                   # clDice / Betti-0 evaluation metrics (hard skeletons)
│   ├── metrics/                 # F1/Se/Sp/AUC-ROC, ECE/Brier/NLL, AUROC-err/AURC
│   ├── inference/               # Sec 3.9 — full-image tiled inference, one global graph
│   ├── engine/                  # 3-stage trainer + evaluator
│   ├── stats/                   # paired bootstrap, Wilcoxon, Holm correction (Sec 4.4)
│   └── utils/                   # config loading, seeding, checkpointing
├── configs/                      # base.yaml + one file per dataset (hyperparameters = Table S1)
├── scripts/
│   ├── train.py                     train one dataset/seed end to end
│   ├── evaluate.py                  evaluate a checkpoint, Section 4.2/4.4 aggregation
│   ├── run_ablation.py              construct one Table 5 ablation variant
│   ├── make_tables.py               regenerate results/tables/ + results/data/*.csv
│   ├── make_figures.py              regenerate results/figures/
│   └── _paper_reported_results.py   the published Table 1-5 numbers (single source of truth)
├── results/
│   ├── data/                        Table 1-5 as CSV
│   ├── tables/                      Table 1-5 as Markdown (+ LaTeX for Table 1)
│   └── figures/                     Fig. 1-8, see below
├── tests/                        35 pytest unit/integration tests (see Testing)
├── requirements.txt / pyproject.toml
└── LICENSE (MIT)
```

Every module's docstring cites the paper section and equation it implements, and states explicitly where it had to make an implementation choice the paper's prose doesn't pin down (e.g. how heterogeneous observed/gap edge features are embedded into one space before the GNN's attention and gate, Eq. 7-8).

---

## Installation

```bash
git clone <this-repo>
cd EviTopo-GNN
pip install -r requirements.txt
# or, for an editable install of the package itself:
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1 (CPU is enough to run the test suite and the tables/figures scripts; GPU is needed for real training — Section 6.3 reports 463 GFLOPs / 61.4 ms per 512×512 tile on an RTX A6000).

## Quickstart

```python
import torch
from evitopo_gnn import EviTopoGNN

model = EviTopoGNN()                 # defaults match Table S1 / configs/base.yaml
model.eval()

x = torch.rand(1, 3, 512, 512)        # a batch of fundus photographs
with torch.no_grad():
    out = model(x, inference=True)    # applies Eq. (11)'s bridge decision

prob = out.final[0].prob              # [1, 1, 512, 512], the vessel probability P
uncertainty = out.final[0].uncertainty  # [1, 1, 512, 512], the uncertainty U
graph = out.graphs[0]                 # the per-image VesselGraph, G
```

For a full-resolution image larger than one training tile, use the tiled inference engine (Section 3.9 — one global graph, not per-tile graphs):

```python
from evitopo_gnn.inference import TiledInferenceEngine

engine = TiledInferenceEngine(model, window=512, stride=256)
result = engine.run(image, fov_mask)   # image: [1, 3, H, W]
```

### Training

```bash
python scripts/train.py --config configs/drive.yaml --seed 0 --output runs/drive_seed0
```

Datasets must already sit under `<dataset.root>/<dataset.name>/{images,manual,mask}/` (see `evitopo_gnn/data/datasets.py`'s docstring) — this script does not download data, since DRIVE/STARE/CHASE_DB1/HRF/FIVES [2–6] are third-party releases with their own licenses.

### Evaluation

```bash
python scripts/evaluate.py --config configs/drive.yaml \
    --checkpoint runs/drive_seed0/best.pt --split test \
    --output results/data/drive_seed0_predictions.json
```

Reports pooled pixel metrics (F1/precision/Se/Sp) and per-image clDice/Betti-0/ECE, matching the two estimands of Section 4.4 — see `evitopo_gnn/engine/evaluator.py`.

### Ablations

```bash
python scripts/run_ablation.py --variant A5a --config configs/drive.yaml --seed 0
```

`scripts/run_ablation.py --help` lists which of Table 5's ~23 rows are directly constructible from this repo's config/constructor flags today (e.g. the gate variants A5/A5a/A5b/A9g via `gnn_gate_mode`) and which name an architectural variant (A3a's kNN graph, A11's Kolmogorov-Arnold node update, …) that would need its own module first — stated explicitly rather than silently approximated.

---

## Results section: tables and figures

All of Section 5–6's tables and figures are regenerated by two scripts from one source of truth, [`scripts/_paper_reported_results.py`](scripts/_paper_reported_results.py) (the published numbers, transcribed from the manuscript):

```bash
python scripts/make_tables.py    # -> results/data/*.csv, results/tables/*.md (+ .tex for Table 1)
python scripts/make_figures.py   # -> results/figures/*.png
```

### Tables

| # | Table | Markdown | CSV |
|---|---|---|---|
| 1 | In-domain comparison, 11 baselines + ours, 5 datasets (F1, clDice) | [table1_in_domain.md](results/tables/table1_in_domain.md) | [csv](results/data/table1_in_domain.csv) |
| 2 | Paired comparison vs. strongest baseline (ΔF1/ΔclDice/ΔECE, 95% CI, Holm-adjusted p) | [table2_paired_stats.md](results/tables/table2_paired_stats.md) | [csv](results/data/table2_paired_stats.csv) |
| 3 | Uncertainty quality on DRIVE vs. MC dropout / deep ensemble | [table3_uncertainty.md](results/tables/table3_uncertainty.md) | [csv](results/data/table3_uncertainty.csv) |
| 4 | Controlled gap test: correct/false reconnections | [table4_reconnections.md](results/tables/table4_reconnections.md) | [csv](results/data/table4_reconnections.csv) |
| 5 | Component ablation (23 rows, DRIVE + FIVES) | [table5_ablation.md](results/tables/table5_ablation.md) | [csv](results/data/table5_ablation.csv) |

### Figures

| # | File | What it shows |
|---|---|---|
| 1 | `fig1_pipeline.png` | The five-stage pipeline (Eq. 1) |
| 2 | `fig2_f1_cldice_by_dataset.png` | Table 1 in full: every method's F1/clDice, every dataset |
| 3 | `fig3_betti0_slopegraph.png` | Table 2's Betti-0 error, strongest baseline → ours |
| 4 | `fig4_paired_delta_forest.png` | Table 2's paired Δ(F1, clDice, ECE) with 95% CI, significance marked |
| 5 | `fig5_uncertainty_quality.png` | Table 3: ECE by pixel region; AUROC-err / AURC |
| 6 | `fig6_reconnections.png` | Table 4: recall, bridging precision, false reconnections/image |
| 7a/b | `fig7a/b_ablation_*.png` | Table 5's ablation sequence (A1→A9): overlap/connectivity, then topology/calibration |
| 8 | `fig8_latency_breakdown.png` | Section 6.3's per-stage tile latency (61.4 ms total) |

Chart construction follows this repo's data-viz conventions: one y-axis per chart (never dual-axis), a fixed, colorblind-validated categorical palette used in a stable slot order, and "ours" picked out against neutral gray rather than relying on a 12-way rainbow legend.

### Reproducing the paper's tables and figures

Be precise about what `make_tables.py` / `make_figures.py` do and don't do: **they format the paper's own published numbers**, they do not re-run an experiment. Turning this repository into something that *regenerates* Table 1–5 from scratch would additionally require:

1. The five datasets under their original licenses (DRIVE [2], STARE [3], CHASE_DB1 [4], HRF [5], FIVES [6]) — not redistributed here.
2. The public RETFound `RETFound_cfp_weights` checkpoint [22] for the frozen ViT-L/16 branch (`evitopo_gnn/models/hybrid_encoder.py::RETFoundViT.load_retfound_checkpoint` loads it once available).
3. Training compute matching Section 3.9 / 6.3: three-stage schedule to 200 epochs, 5 seeds × 5 datasets (STARE and HRF additionally need their leave-one-out / k-fold protocols run end to end), on hardware comparable to the paper's RTX A6000.
4. Running `scripts/train.py` + `scripts/evaluate.py` per (dataset, seed), then `evitopo_gnn/stats/significance.py` to recompute Table 2's paired bootstrap/Wilcoxon/Holm statistics from the resulting per-image metrics, in place of the transcribed values in `_paper_reported_results.py`.

None of that is available in the environment this repository was built in; the architecture, losses, graph construction, metrics, statistics and training/evaluation loop are nonetheless implemented end to end and covered by the test suite below, so step 4 is "point it at real data and let it run," not "write the code."

---

## Testing

```bash
pip install -r requirements.txt pytest
pytest tests/ -q
```

35 tests across `tests/test_model_forward.py` (forward/backward shapes, confirms gradient reaches the frozen ViT backbone only through its LoRA adapters), `test_graph_builder.py`, `test_losses.py`, `test_metrics.py`, `test_stats.py`, `test_data.py`, `test_datasets.py` and `test_engine.py` (the 3-stage trainer and tiled inference end to end on tiny configs).

---

## Method → code map

| Paper | Section / Eq. | Code |
|---|---|---|
| Problem formulation | Eq. 1 | `evitopo_gnn/models/evitopo_gnn.py` |
| LoRA adapters | Eq. 2 | `evitopo_gnn/models/lora.py` |
| Hybrid ViT-CNN encoder, cross-attention fusion | 3.2, Eq. 3 | `evitopo_gnn/models/hybrid_encoder.py` |
| Evidential coarse/final heads | 3.3, Eq. 4-5 | `evitopo_gnn/models/evidential_head.py`, `evitopo_gnn/losses/evidential_loss.py` |
| Vessel-topology graph + gap candidates | 3.4, Eq. 6 | `evitopo_gnn/models/graph_builder.py` |
| Uncertainty-gated GNN | 3.5, Eq. 7-9 | `evitopo_gnn/models/gnn_layers.py` |
| Evidential gap-link head | 3.6, Eq. 10-11 | `evitopo_gnn/models/link_predictor.py`, `evitopo_gnn/losses/link_loss.py` |
| Graph-to-pixel decoder | 3.7, Eq. 12 | `evitopo_gnn/models/decoder.py` |
| Topology-aware contrastive loss | 3.8, Eq. 13-14 | `evitopo_gnn/losses/contrastive_loss.py` |
| Objective, 3-stage schedule | 3.9, Eq. 15 | `evitopo_gnn/losses/combined_objective.py`, `evitopo_gnn/engine/trainer.py` |
| Full-image tiled inference | 3.9 | `evitopo_gnn/inference/tiled_inference.py` |
| Datasets, splits, leakage control | 4.1 | `evitopo_gnn/data/` |
| Metrics (F1, clDice, Betti-0, ECE, AUROC-err, AURC) | 4.2 | `evitopo_gnn/graph/topology_metrics.py`, `evitopo_gnn/metrics/` |
| Model selection & statistics | 4.4 | `evitopo_gnn/stats/significance.py` |
| Synthetic gap augmentation | 3.6 | `evitopo_gnn/data/synthetic_gaps.py` |

## Citation

```bibtex
@article{fatima2026evitopognn,
  title   = {EviTopo-GNN: Evidential Vessel-Topology Graph Learning with
             Uncertainty-Guided Contrastive Representation for Retinal
             Vessel Segmentation in Diabetic Retinopathy},
  author  = {Fatima, Iram and Khan, Ihtiram Raza and Kaur, Harleen},
  journal = {Manuscript},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The five benchmark datasets (DRIVE, STARE, CHASE_DB1, HRF, FIVES) and the RETFound checkpoint are third-party releases under their own licenses and are not redistributed here.
