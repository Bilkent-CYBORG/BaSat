# Source traceability

BayOPy is a clean-room implementation. The implementation guide and the papers
below govern mathematical behavior. Referenced repositories are used only to
construct independent numerical regression fixtures. No repository source is
vendored or translated line-by-line.

| Algorithm | Governing publication | Repository status | BayOPy boundary |
|---|---|---|---|
| REMBO | [Wang et al., *Bayesian Optimization in a Billion Dimensions via Random Embeddings* (2016)](https://doi.org/10.1613/jair.4806) | [Author repository](https://github.com/ziyuw/rembo) | Ask-tell, Torch GP, finite-data and q extensions are BayOPy additions. |
| HeSBO | [Nayebi, Munteanu, and Poloczek, *A Framework for Bayesian Optimization in Embedded Subspaces* (2019)](https://proceedings.mlr.press/v97/nayebi19a.html) | [Author repository](https://github.com/aminnayebi/HesBO) | Exact hash/sign inverse only; no invented pseudoinverse. |
| ALEBO | [Letham et al., *Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization* (2020)](https://proceedings.neurips.cc/paper/2020/hash/10fb6cfa4c990d2bad5ddef4f70e8ba2-Abstract.html) | [Official repository](https://github.com/facebookresearch/alebo) and Ax | MAP mode and finite-row selection are explicit extensions. |
| ADD-GP-UCB | [Kandasamy et al., *High Dimensional Bayesian Optimisation and Bandits via Additive Models* (2015)](https://proceedings.mlr.press/v37/kandasamy15.html) | [Author repository](https://github.com/kirthevasank/add-gp-bandits) | Full-row finite acquisition prevents fabricated Cartesian actions. |
| TuRBO | [Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization* (2019)](https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html) | [Official repository](https://github.com/uber-research/TuRBO) | First release implements TuRBO-1; finite-row filtering is an extension. |
| Dropout | [Li et al., *High Dimensional Bayesian Optimization Using Dropout* (2017)](https://www.ijcai.org/proceedings/2017/0291.pdf) | [Regression repository](https://github.com/zhandawei/Bayesian_Optimization_Algorithms) | Finite-row behavior is an extension. |
| MCTS-VS | [Song et al., *Monte Carlo Tree Search based Variable Selection for High Dimensional Bayesian Optimization* (2022)](https://papers.nips.cc/paper_files/paper/2022/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html) | [Official repository](https://github.com/lamda-bbo/MCTS-VS) | Deterministic BayOPy defaults are identified in configuration docs. |
| SAASBO | [Eriksson and Jankowiak, *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces* (2021)](https://proceedings.mlr.press/v161/eriksson21a.html) | [Author repository](https://github.com/martinjankowiak/saasbo) | Requires the `mcmc` capability. |
| AG-VS | Bayburt and Tekin, *Adaptive Group Variable Selection for High-Dimensional Bayesian Optimization* (ICLR 2027 submission) | Local governing manuscript and `/Users/denizc/Desktop/AG_VS` reference | Eq. (8) is unnormalized; node GPs use the reference isotropic Matérn-5/2 covariance and seeded random splitting rather than ARD selection; finite-row and joint-q selection are explicit BayOPy adaptations. |
| LA-MCTS | [Wang, Fonseca, and Tian, *Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search* (2020)](https://proceedings.neurips.cc/paper/2020/hash/e2ce14e81dba66dbff9cbc35ecfdb704-Abstract.html) | [Official repository](https://github.com/facebookresearch/LA-MCTS) | Paper-general preset is the default; other presets are named. |
| PG, EG, GS, STS | [Cai, Gomes, and Scarlett, *Lenient Regret and Good-Action Identification in Gaussian Process Bandits* (2021)](https://proceedings.mlr.press/v139/cai21c.html) | [Author repository](https://github.com/caitree/GoodAction) | Version 1 supports the paper's maximization threshold convention. |
| StableOPT | [Bogunovic et al., *Adversarially Robust Optimization with Gaussian Processes* (2018)](https://arxiv.org/abs/1810.10775) | Paper governs | External adversarial composition is reported as an extension. |
| AdveRS | [Saday, Yildirim, and Tekin, *Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks* (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4830a9b95a2f63fc4b3fe09abc18f045-Abstract-Conference.html) | [Official repository](https://github.com/Bilkent-CYBORG/AdveRS) | One class exposes RS1, RS2, and RSG variants. |
| CGP-UCB | [Krause and Ong, *Contextual Gaussian Process Bandit Optimization* (2011)](https://proceedings.neurips.cc/paper/2011/hash/f3f1b7fc5a8779a9e618e1f23a7b7860-Abstract.html) | Paper governs; community code is not an authority | Joint action-context coordinates are explicit. |
| RoBOS | [Saday, Yildirim, and Tekin, *Robust Bayesian Satisficing* (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/daa098aa8e1fc718943ff1ab7b5b30c9-Abstract-Conference.html) | [Official repository](https://github.com/Bilkent-CYBORG/RoBOS) | Finite data mode requires complete context coverage. |
| DRBO | [Kirschner et al., *Distributionally Robust Bayesian Optimization* (2020)](https://proceedings.mlr.press/v108/kirschner20a.html) | Paper governs; RoBOS is related only | General finite-context MMD protocol only; simulator mode is excluded. |

## Source-guarantee reporting

Every algorithm exposes a `SourceRecord`. Diagnostics set
`source_guarantees_applicable=False` when the selected mode uses a non-native
environment, data-backed adaptation, generic q extension, approximate solver,
custom kernel, or deterministic BayOPy default that changes source behavior.
The associated reason code is retained in diagnostics and result metadata.

BayOPy does not bundle benchmark recipes or label application runs as paper
reproductions. Users must document the relationship between their own objective,
dataset, numerical budget, and the corresponding publication experiment.
