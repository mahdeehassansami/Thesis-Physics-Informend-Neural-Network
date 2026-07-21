# Publication novelty-threat matrix

Status: provisional research-design record

Search cut-off: 21 July 2026

Evidence base: local papers and reference implementations, official dataset documentation,
publisher pages, and primary papers indexed during the current search.

## Purpose and limitation

This matrix defines what the project may and may not currently claim as novel. It is not a
completed systematic review and must be refreshed before manuscript submission. A statement
that no prior work was located means exactly that; it is not proof that no such work exists.

The starting evidence is the independently verified Run 4/Run 5 pair. Run 5 reduced the mean
label-free feature-distribution discrepancy in every IMS fold, but Weak-PINN RUL error improved
on only two of four held-out bearings and late-life MAE worsened. The fixed strong-physics model
also remained worse than Weak-PINN. These observations motivate a study of assumption and
physics-prior validity; they do not demonstrate a new method.

## Closest prior work

| Work | What it already contributes | Evidence/protocol | Consequence for this project |
|---|---|---|---|
| Liao et al., *Advanced Engineering Informatics* 58 (2023), [DOI 10.1016/j.aei.2023.102195](https://doi.org/10.1016/j.aei.2023.102195) | Self-attention-assisted PINN, DeepHPM latent differential operator, and ReLoBRaLo adaptive data/physics loss balancing | C-MAPSS FD004 | Attention, a learned hidden operator, or generic adaptive loss weighting is not novel. The upstream code is an aircraft-engine reference, not a bearing-physics implementation. |
| Hu et al., *Journal of Advanced Manufacturing Science and Technology* 4 (2024), [DOI 10.51393/j.jamst.2024018](https://doi.org/10.51393/j.jamst.2024018) | DSCN-AttnPINN bearing adaptation with a DeepHPM implicit operator | XJTU-SY condition 1, leave-one-bearing-out | AttnPINN applied to bearings is already published. Its learned operator is latent physics, not an identified rolling-contact law. |
| Chen et al., *Journal of Dynamics, Monitoring and Diagnostics* 1 (2022), local paper `Physics-Informed Deep Neural Network for Bearing Prognosis with Multisensory Signals.pdf` | Multisensory vibration/temperature bearing prognosis with physics-informed degradation constraints | Laboratory bearing data | Vibration/temperature fusion and weak trend constraints are not sufficient novelty. |
| Jiang et al., *Advanced Engineering Informatics* 63 (2025), [DOI 10.1016/j.aei.2024.102958](https://doi.org/10.1016/j.aei.2024.102958) | Spatio-temporal attention and hidden-physics RUL modeling | Machinery RUL benchmarks | A new attention block around DeepHPM is crowded. |
| Lv et al., *Machines* 13 (2025), [DOI 10.3390/machines13060452](https://doi.org/10.3390/machines13060452) | SiMBA-PINN, latent hidden physics, and dynamic gating/fusion | C-MAPSS | A generic data/physics gate is not by itself novel. |
| Dhibi et al., *Integrating Materials and Manufacturing Innovation* 15 (2026), [DOI 10.1007/s40192-026-00441-w](https://doi.org/10.1007/s40192-026-00441-w) | Bearing PI-BiLSTM with GP-estimated sequence-specific Paris parameters, Huber residual, curriculum, Bayesian optimization of physics weight, deep ensembles, optional split conformal calibration, and a Paris-consistency score | PRONOSTIA | This directly rules out adaptive Paris parameters, curriculum weighting, physics consistency, ensembles, or conformal intervals as standalone novelty. It assumes a Paris-shaped decay of normalized RUL and evaluates consistency with that same assumed law; it does not establish whether the law is applicable. |
| Wang et al., *Reliability Engineering & System Safety* 266 (2026), [DOI 10.1016/j.ress.2025.111778](https://doi.org/10.1016/j.ress.2025.111778) | Physics-constrained Bayesian network with Weibull time-to-failure likelihood and predictive uncertainty | Turbofan and bearing data | Bayesian PINN plus uncertainty is already occupied. |
| Lei et al., *Engineering Research Express* 8 (2026), [DOI 10.1088/2631-8695/ae3274](https://doi.org/10.1088/2631-8695/ae3274) | Multi-scale attention PINN with learned nonlinear operator, Monte Carlo sampling, KDE, and probabilistic RUL intervals | Slurry pump and C-MAPSS | Multi-scale attention plus hidden physics plus UQ is not a defensible central claim. |
| Daniels et al., ICLR 2026, [OpenReview](https://openreview.net/forum?id=7PORoDlSS4) | Physics-Informed Log Evidence (PILE), an uncertainty-aware model-selection diagnostic for GP-based physics-informed learning | PDE/kernel benchmarks | “Physics-aware model selection” is broader than bearing RUL. PILE should be a conceptual or implemented baseline where feasible; the proposed work must address prior applicability under bearing degradation, not merely select a scalar weight. |
| Dhibi et al. 2026 and PHM Europe 2026 bearing conformal work, [DOI 10.36001/phme.2026.v9i1.4902](https://doi.org/10.36001/phme.2026.v9i1.4902) | Ensemble/conformal bearing RUL intervals and empirical calibration | PRONOSTIA or XJTU-SY | Conformal prediction is a secondary reliability analysis only, not the novelty claim. Conditional coverage under domain shift must be reported rather than implying a marginal guarantee solves it. |
| MDDG, *Measurement* 249 (2025), [DOI 10.1016/j.measurement.2024.116451](https://doi.org/10.1016/j.measurement.2024.116451) | Multi-domain degradation feature generalization to unseen bearing domains | Two bearing datasets | Domain generalization alone is crowded. A representative domain-generalization baseline is required. |
| TG-PITL, *Reliability Engineering & System Safety* (2026), [publisher page](https://www.sciencedirect.com/science/article/pii/S0951832025013572) | Trend-guided physics-informed transfer for cross-machine bearing RUL | Cross-machine bearing transfer | Cross-machine physics-informed transfer is already claimed. Our method must test whether a prior should transfer, not simply transfer it. |
| Domain-adaptive PINN with adaptive weighting, 2025 preprint, [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5351528) | Feature/physics transfer with adaptive weighting | Preprint evidence | Even before peer review, it is a novelty threat to generic adaptive domain/physics weighting. |
| Herwig et al., *Mechanical Systems and Signal Processing* 226 (2025), [DOI 10.1016/j.ymssp.2025.112925](https://doi.org/10.1016/j.ymssp.2025.112925) | BearingNet uses physically bounded analytical signal-processing layers for interpretable and robust diagnosis | Simulated and real diagnostic datasets | It supports using explicit, testable signal physics, but it is diagnosis rather than RUL. It is a methodological precedent, not a direct competitor. |
| Holistic bearing simulation model, ESREL-SRA-E 2025, [official proceedings](https://rpsonline.com.sg/proceedings/esrel-sra-e2025/html/ESREL-SRA-E2025-P8028.html) | Synthetic full-life bearing trajectories with controlled load, life, degradation-family, dynamics, slip, and noise parameters | 28 training and 12 test trajectories in the supplied v2 dataset | This enables known-truth prior-validity stress tests. It must not be presented as real-world validation. |

## Claims ruled out by the audit

The following are not acceptable central novelty claims:

- “The first AttnPINN for bearing RUL.”
- “The first PINN to combine attention and bearing RUL.”
- “The first adaptive weighting of data and physics losses.”
- “The first adaptive or sequence-specific Paris-law bearing RUL model.”
- “The first physics-consistency score for bearing RUL.”
- “The first Bayesian/ensemble/conformal PINN for probabilistic RUL.”
- “The first physics-informed cross-domain or cross-machine bearing RUL model.”
- “A physically interpretable model” when the only physics is an unconstrained DeepHPM
  operator learned from labels.
- “Actual bearing physics” when unmeasured load, crack geometry, lubricant state, or material
  parameters have been inserted as fixed constants.

## Provisional open gap

No reviewed bearing-RUL paper was found that combines all three of the following:

1. Treats the *applicability* of each candidate degradation prior as an explicit quantity,
   distinct from predictive uncertainty, residual size, or an optimized loss weight.
2. Validates that quantity against known-valid and deliberately misspecified physics before
   using it to control physics regularization.
3. Measures whether the mechanism prevents physics-induced negative transfer on unseen real
   bearings and operating conditions while retaining a declared data-only fallback.

This gap is narrower than “adaptive physics.” Dhibi et al. adapt Paris parameters but retain
the Paris-shaped law. PILE selects physics-informed model settings in a GP/PDE setting. Generic
adaptive weighting balances objectives. The proposed question is instead: *Is this law a
credible prior for this bearing, lifecycle region, and observed operating context at all?*

## Working novelty hypothesis

A **physics-prior credibility mechanism**, trained and calibrated without target-test RUL
labels, can identify when a candidate bearing-degradation law is misspecified and reduce the
negative-transfer regret of physics-informed RUL prediction compared with fixed, validation-
selected, and generic adaptive loss weighting.

The potentially publishable contribution is the combination of:

- a known-truth physics-applicability stress-test benchmark;
- an anti-collapse, label-safe credibility estimator for multiple candidate priors;
- a data-only fallback when every strong prior is unsupported; and
- controlled synthetic-to-real evidence about negative transfer, not merely a lower average
  RMSE on one benchmark.

This remains a hypothesis until the protocol in `PUBLICATION_PROTOCOL.md` is executed. The
project must not use “first,” “novel,” “trustworthy,” or “high-impact” as a demonstrated result
before the full literature refresh and experiments succeed.

## Search refresh triggers

Repeat and extend the search:

- immediately before implementing the final method;
- before choosing manuscript title and contribution language;
- before submission; and
- whenever a 2026 paper adds physics applicability, model-discrepancy detection, bearing-RUL
  negative-transfer prevention, or conditional reliability under regime shift.

At each refresh, record query strings, databases, dates, inclusion/exclusion decisions, and
new threats. Include IEEE Xplore, Scopus or Web of Science, ScienceDirect, SpringerLink, IOP,
Crossref, Google Scholar citation chaining, arXiv, and major PHM conference proceedings.
