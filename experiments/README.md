# Experiment artefacts

Everything here is output. `scripts/run_experiment.py` writes the grids,
`scripts/analyse.py` writes the report and the figures. Nothing in this
directory is edited by hand, and nothing here is an input to the pipeline.

## Results grids

Three CSVs, because two of them are superseded and are kept deliberately rather
than left behind.

| File | Rows | Products | What it is |
|---|---|---|---|
| `results.csv` | 5,184 | 24 | The current grid. Everything reported comes from this file. |
| `results_prev.csv` | 3,600 | 20 | The grid as it stood before the dataset grew to 24 and before the generative relighting arm was added. Superseded; kept so the expansion can be seen rather than asserted. |
| `results_broken_metric.csv` | 2,400 | 20 | The run made under the first version of `lighting_coherence`, which estimated the product's shading direction by averaging the gradient over the region and so measured the boundary instead. Kept because the write-up reports the correction, and a correction whose artefact has been deleted cannot be checked. |

Do not pool them. The three differ in dataset size, in the treatments present,
and in the case of the last one in what the lighting metric was measuring.

## Report and figures

| File | Produced by |
|---|---|
| `analysis.txt` | `python scripts/analyse.py`, stdout redirected |
| `figures/tradeoff.png` | the fidelity-against-integration curve |
| `figures/ablation.png` | the component ablation |

`analysis.txt` is the file to quote from. The paired contrasts, effect sizes and
proportions of improved pairs are there; the aggregate means available elsewhere
understate the effects, because between-product variance is far larger than any
treatment moves.

## Run scripts

The two sweeps that produced the grids, kept as a record of the exact commands
rather than as a supported interface:

| File | Ran |
|---|---|
| `run_all.sh` | The ablation arms across three seeds, then the relighting sweep, then the analysis |
| `relight_arms.sh` | The four relighting modes including the generative arm, then the ablation at zero relight |

## Logs

One per sweep. The names are the ones the runs were given at the time and are
not systematic; what identifies each is the state it resumed from.

| File | Resumed from | Ran for |
|---|---|---|
| `run.log` | 300 rows | 791 min, the first full sweep |
| `gen.log` | 3,657 rows | 84 min, the generative relighting arm |
| `rerun1.log` | nothing | 25 min, abandoned partway |
| `rerun.log` | 1,059 rows | 17 min |
| `rerun3.log` | 2,160 rows | 19 min |
| `ablation.log` | 1,800 rows | 17 min, the six ablation arms |
| `tr.log` | 4,320 rows | 8 min, the transparent stratum |
| `fill.log` | 3,963 rows | 5 min, filling the last gaps |

They are kept for those resume records. The grid takes hours, the runner
resumes at row granularity, and these lines are the evidence that it does: the
first sweep in `run.log` was interrupted by the machine suspending and picked up
where it left off. Nothing here is needed to reproduce the results, which come
from `results.csv`.

The progress bars have been stripped. They were roughly two thirds of every
file, they carried nothing the surrounding lines do not, and as
carriage-return output redirected to a file they rendered as mojibake.

## Not in the repository

`samples/` holds the pre-rendered stimuli the perceptual study draws from,
around 1.6 GB, and `human_study.csv` holds any responses collected. Both are
gitignored. Regenerate the stimuli with:

```bash
python scripts/run_experiment.py --variants all
```
