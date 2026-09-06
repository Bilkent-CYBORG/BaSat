# Algorithm and environment compatibility

The executable source of truth is
`tests/integration/test_executable_contract_matrix.py`. It constructs all
`19 x 4 x 2 = 152` algorithm/environment/source-mode cases through
`Experiment`. Every case has one explicit expected outcome: `native`,
`extension`, `adapted`, `cross_environment`, or `incompatible`. The locked
default-policy inventory contains 19 native, 21 same-family extensions, 16
adapters, 88 directly executable cross-environment cases, and 8 incompatible
cases for the matrix's standard configurations.

| Algorithm | Function-backed native environment | Data-backed requirement |
|---|---|---|
| REMBO | Stochastic | Identical stored embedding fingerprint and explicit low-dimensional coordinate. |
| HeSBO | Stochastic | Exact hash-bucket-consistent inverse or explicit embedded coordinate. |
| ALEBO | Stochastic | Exact row-space member satisfying the embedding polytope. |
| ADD-GP-UCB | Stochastic | Score complete available rows; never combine independent group optima. |
| TuRBO | Stochastic | Filter available rows to the active trust region; restart or exhaust explicitly. |
| Dropout | Stochastic | Score projections of complete rows; filling cannot synthesize an action. |
| MCTS-VS | Stochastic | Nested optimizer scores complete rows under the selected variable subset. |
| SAASBO | Stochastic | Fully Bayesian acquisition is evaluated only on available rows. |
| AG-VS | Stochastic | Complete available rows are scored through the active coordinate projection; no synthetic finite row is constructed. |
| LA-MCTS | Stochastic | Filter rows through the complete root-to-leaf classifier path. |
| PG | Stochastic | Probability of goodness is scored on available rows. |
| EG | Stochastic | Fixed-threshold expected improvement is scored on available rows. |
| GS | Stochastic | Fantasies use learner-visible data; candidate rows never reveal hidden outcomes. |
| Satisficing-TS | Stochastic | One joint sample covers all available rows; stable closest-good selection. |
| StableOPT | Stochastic; adversarial is an extension | Every robust-neighborhood query must be an explicitly stored row. |
| AdveRS | Adversarial | Inner perturbation neighborhoods must be explicit finite sets. |
| CGP-UCB | Contextual | Score only rows matching the current revealed context. |
| RoBOS | Distributional | Each exposed action has a complete row for every context in support. |
| DRBO | Distributional | Each exposed action has a complete row for every context in support. |

Unlisted pairings are diagnosed from their learner-visible contracts. If all
required fields are already present, they run as `cross_environment`; the
family mismatch alone is never an error. CGP-UCB uses a constant-context adapter
in a stochastic environment and the current reference distribution's weighted
support mean as its context in a distributional environment. DRBO and RoBOS use a constant-context,
point-mass-distribution adapter in every non-distributional environment. The
reference is `[1]` for one configured context or a one-hot vector on the first
configured context for a larger existing support; the margin is `0`. A
contextual source continues evaluating its real context, but the adapted
distributional learner deliberately sees only the constant context. These runs
emit `AdapterAppliedWarning` and `AssumptionViolationWarning` and disable source
guarantees. Only malformed data, inconsistent user-supplied support, exhausted
finite candidates, or unusable numerical configurations remain errors.

## Additional axes

The 152 contract cases are augmented with pairwise cases for batch behavior,
direction, historical initialization, noise semantics, duplicate/exhausted data,
checkpoint position, dtype/device, and `D_env != D_model`. Continuous numerical
inputs are property-tested rather than represented as an impossible literal
Cartesian product.
