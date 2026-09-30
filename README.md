# EdgeTrust Performance Evaluation Pipeline

Real, runnable implementation of the EdgeTrust six-equation trust core
(Eqs. 1-6), the LightGBM Layer-1 gate, and the full Performance
Evaluation section (Scenarios A-F), meant to run against the actual
CICIoT2023 dataset.

## What this is, and isn't

This is real code that computes real metrics from whatever data you
give it. It has been syntax-checked and run end-to-end against a small
synthetic CSV built to match CICIoT2023's real column schema, to catch
plumbing bugs before you ever see them, not against the real dataset
(I don't have network access to unb.ca in the environment that built
this). Numbers you get out of it the first time you run it against your
real data are the first real numbers this pipeline has ever produced.
Expect to debug a little, exactly as you would with any new codebase,
particularly around `config.LABEL_TO_CATEGORY` if your CSV release uses
slightly different label strings than the ones assumed here (the loader
prints every label it can't match, rather than failing silently).

## Setup

```bash
pip install -r requirements.txt
```

Place the CICIoT2023 CSV part-files (or a subset of them) under
`data/CICIoT2023/` (subdirectories are fine, the loader searches
recursively).

## Running it

Start small. The full release is ~47 million rows across ~169 files;
don't run the whole thing until a subsample has already run cleanly:

```bash
python run_all.py --sample-frac 0.02          # ~2% of each file, fast iteration
python run_all.py --max-rows-per-file 50000   # cap per file instead
python run_all.py                              # full dataset, once you're ready
```

## What it produces

- `outputs/figures/` -- every figure in the Performance Evaluation
  section: confusion matrix, ROC curve, per-category recall, ablation
  comparison, attack-resilience trajectories, throughput projection,
  Theorem 5 verification, baseline comparison.
- `outputs/tables/` -- the same results as CSV/JSON, for pasting
  directly into manuscript tables.
- `outputs/report.md` -- a single auto-generated summary with every
  scenario's headline numbers.

## Known limitations, stated plainly

1. **No device/session ID in the public CSV release.** The manuscript
   calls for device/session-aware temporal splitting; this pipeline
   falls back to stratified random splitting and says so loudly
   (`data_pipeline.py`, `train_val_test_split`). Scenario C's
   "attack-resilience sessions" are a best-effort proxy built from
   contiguous post-split row blocks, not genuine per-device traces.
   If you have the original device-to-flow mapping, replace this
   splitting logic with a real group-aware split.
2. **Feature direction assumptions.** Eq. (1)'s benefit/cost mapping
   needs a direction per feature; `data_pipeline.FEATURE_DIRECTION_OVERRIDE`
   defaults every selected feature to "cost" (higher = more suspicious).
   Confirm this matches your own EDA before trusting the entropy weights.
3. **Latency/throughput are only as real as the hardware you run this
   on.** Run `run_all.py` on the actual target hardware (e.g., the
   Raspberry Pi 4B) if the resulting numbers are meant to support a
   hardware latency claim; numbers from a dev laptop are not that.
4. **Scenario F's baseline set is a starting point**, not the full
   "fifteen competitive baselines" mentioned in the Abstract. Add more
   candidates in `scenarios.run_baseline_comparison` as needed.
5. Uses `delta=0.08`, `m=14`, matching the manuscript's current
   parameter block. If you've applied the earlier hysteresis fix
   (delta=0.05, 3-epoch promotion confirmation), update `config.py`;
   the promotion-confirmation logic is already implemented in
   `trust_engine.EdgeTrustEngine`.

## File map

```
config.py               all Eq.(1)-(6) parameters + label taxonomy + feature list
src/data_pipeline.py    loading, label mapping, feature selection, splitting, Eq.(1) normalization
src/entropy_weights.py  Eq.(1)-(2) Shannon entropy weights + Theorem 1 check
src/trust_engine.py     Layer-1 LightGBM wrapper + Eqs.(3)-(6) + ABAC-lite decision mapping
src/metrics.py          confusion matrix family, per-category recall, state-flap rate,
                        corrected Theorem 5 bound check, latency/throughput measurement
src/scenarios.py        ablation (B), attack resilience (C), baseline comparison (F)
src/plotting.py         every figure
run_all.py              orchestrates everything end to end
```
