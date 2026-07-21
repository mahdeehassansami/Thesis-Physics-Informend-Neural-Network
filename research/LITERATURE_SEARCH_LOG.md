# Literature search log

## 21 July 2026 — publication-direction audit

### Question

Which parts of the proposed bearing-RUL work remain plausibly novel after accounting for
current physics-informed, domain-generalization, and uncertainty-aware prognostics research?

### Sources inspected

- All PDFs under `papers/` relevant to bearing RUL, PINNs, DeepHPM, fatigue/crack growth,
  multisensory prognosis, domain adaptation, and evaluation.
- `Instructions.pdf` and dataset documentation for IMS, PRONOSTIA, the vibration/temperature
  run, and Bearings with Varying Degradation Behaviors v2.
- Source and README files in `AttnPINN-for-RUL-Estimation-English/`,
  `physics-informed-neural-network-main/`, and `Bearing_Simulation_Model-main/`.
- Primary publisher or official pages from ScienceDirect, SpringerLink, IOP, MDPI, IEEE/PHM
  proceedings, OpenReview, SSRN, and arXiv for 2021–2026 work.

### Query families

- bearing remaining useful life physics-informed neural network;
- bearing RUL Paris law adaptive regularization;
- self-attention / hidden physics / DeepHPM RUL;
- physics-informed RUL adaptive loss weighting;
- bearing RUL domain generalization and cross-machine transfer;
- bearing RUL uncertainty quantification, Bayesian, ensemble, and conformal prediction;
- physics prior credibility, model discrepancy, misspecified physics, and negative transfer;
- physics-informed diagnostics and model evidence; and
- controlled physics-informed simulation under unseen operating conditions.

Exact strings used in the final focused search included:

- `"Remaining useful life prediction with uncertainty quantification" "physics-informed" bearing`
- `"multi-scale attention-based physics-informed neural network" RUL uncertainty`
- `physics prior credibility model discrepancy prognostics remaining useful life`
- `bearing RUL physics misspecification negative transfer prior credibility`
- `"Uncertainty-Aware Diagnostics for Physics-Informed"`
- `"physics prior" credibility "remaining useful life"`
- `"physics-informed" "negative transfer" prognostics`

### Inclusion logic

Included primary work that contributes at least one of:

- a physics-informed RUL mechanism;
- bearing RUL under domain or operating-condition shift;
- explicit degradation/crack/life laws;
- adaptive physics weighting or parameter identification;
- physics diagnostic/model-selection evidence; or
- predictive uncertainty/calibration for RUL.

Reviews were used to find primary papers and frame field-level challenges, not to establish
method details when a primary source was available. Preprints were retained as novelty threats
but labeled as preprints. Diagnostic-only bearing work was retained only where it established a
relevant physics-embedding precedent.

### Main result

The audit ruled out generic attention, DeepHPM, adaptive weighting, adaptive Paris parameters,
physics-consistency scores, UQ, conformal calibration, and cross-domain transfer as sufficient
novelty. The remaining provisional gap is explicit detection of *prior applicability*, validated
against known-valid and corrupted physics and then used to prevent physics-induced negative
transfer with a declared fallback.

See `PUBLICATION_NOVELTY_MATRIX.md` for the work-by-work record and
`PUBLICATION_PROTOCOL.md` for the resulting falsifiable study.

### Known limitations and required refresh

This was a focused novelty audit, not a PRISMA systematic review. Subscription indexing,
non-English literature, papers not yet indexed, and work published after the cut-off may change
the gap. Before submission, run a documented Scopus/Web of Science/IEEE Xplore search, backward
and forward citation chaining, duplicate removal, and a title/abstract/full-text screening log.
