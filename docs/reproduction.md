# Using the guide

Open [basat_user_guide.ipynb](basat_user_guide.ipynb) after installing the project:

```bash
python -m pip install .
python -m basat notebook docs/basat_user_guide.ipynb
```

The notebook includes its saved plots, including six paper-versus-measurement
comparisons. MHC shows maximum regret only. Viewing these figures does not require
the original experiment archives. Small CSV examples and their provenance are
included in `data/`.

## Running new experiments

The general tutorial cells create new result directories. The reproduction cells
retain their full scientific budgets and may take substantial time. Downloaded
benchmark data and new checkpoints are written locally; they are not bundled.
The saved figures do not imply that a new run will reproduce historical curves
or agree bitwise across machines.

Completed-result reuse is intended for an available, matching archive. The clean
release does not contain those archives. For a new campaign, set
`READ_COMPLETED_RESULTS = False` where present and select a fresh output directory.
Do not disable protocol or runtime compatibility checks to resume old checkpoints.

AdveRS source-mode experiments need a separate Python 3.11 environment:

```bash
python3.11 -m venv .venv-advers
.venv-advers/bin/python -m pip install '.[advers-source]'
.venv-advers/bin/python -m ipykernel install --user --name basat-advers
```

Choose that kernel for the AdveRS reproduction cell. The `advers-source` and
`numpyro` extras have incompatible numerical requirements and must not be
installed together. The ordinary environment can use `.[numpyro]` for SAASBO.

RoBOS source-mode experiments also require a separate Python 3.11 environment
with NumPy 2.4.6 and SciPy 1.15.3:

```bash
python3.11 -m venv .venv-robos
.venv-robos/bin/python -m pip install . 'numpy==2.4.6' 'scipy==1.15.3'
.venv-robos/bin/python -m ipykernel install --user --name basat-robos
```

These are source-compatible numerical environments, not recovered author
environments. See the guards in the experiment cells for additional checks.

Local campaign audits, temporary notebook generators, manuscript sources, and
raw results remain outside this release.
