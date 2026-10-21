# Four external datasets

These files are example data outside the installed `basat` package.

| Environment | Current table | Actions |
|---|---|---:|
| Stochastic | `stochastic/diabetes_bandit.csv` | 442 |
| Contextual | `contextual/wine_bandit.csv` | 178 |
| Distributional | `distributional/wind_power_bandit.csv` | 101 |
| Adversarial | `adversarial/iris_bandit.csv` | 143 |

Load a table with `basat.load_dataset(path, environment="distributional")`,
using the appropriate environment family. `native_manifest.json` records the
schema, transformation, source URLs, and hashes. `manifest.json` identifies
source copies. The notebook contains the environment setup and editable examples.

## Distributional wind-power example

Source: Open Power System Data (2019), Time Series package, version 2019-06-05:
https://data.open-power-system-data.org/time_series/2019-06-05/
Primary German wind data are from the German transmission system operators.
The retained CSV is an exact column/row subset of the hourly download, containing
98 consecutive observations starting 2013-01-01 00:00 UTC.

DRBO Section 4 uses this data package for energy commitment:
https://proceedings.mlr.press/v108/kirschner20a/kirschner20a.pdf

`opsd_wind_2013.csv` retains timestamps, actual generation, installed capacity,
and recorded capacity factor. `wind_power_bandit.csv` is a complete 101-action
by 11-context table of the paper's modeled revenue:

    0.1 * max(c - x, 0) + min(x, c) - 5 * max(x - c, 0)

Action x is promised energy as a fraction of capacity. Context c is recorded
capacity factor rounded to the nearest 0.1. Revenue is calculated from this
explicit model, not a measured market payment. No missing reward is inferred
by the library. `wind_power_provenance.json` records the construction in detail.

The first 48 observations form public history; the next 50 are a chronological
replay. The notebook updates public reference probabilities from the previous
48 observations and supplies each next realization privately to the environment.
All algorithms see the same trace, with 5 independent algorithm seeds and 50
evaluations. Repeated commitments are allowed. The margin 0.1 is illustrative.
This reduced replay is not a reproduction of the paper's annual simulator run.

The diabetes, wine, and iris source copies come from scikit-learn 1.6.1, with
source URLs recorded in the manifests. Their constructed optimization meanings
remain documented in the notebook. Historical experiment archives and checkpoints
are not included in the clean release.
