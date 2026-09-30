# Page classifier models

`page_classifier_v1.json` is the S1 page classifier used on CPU and as the heuristic half of the
GPU classifier: an L2-regularised multinomial logistic regression over the features in
`archrender.understand.features` (visual, text and metadata), with temperature scaling bounded to
T ≥ 1 (softening only: the synthetic calibration set is nearly separable, see DECISIONS.md).
It also stores each feature's training range: pages outside it are sent to review.

- **Training data: synthetic only** (`archrender.synth`), no client or third-party documents
  (owner answer Q-4). The file records the corpus seeds and sizes, the OCR engine version, the git
  commit and the calibration metrics.
- **Retrain** (deterministic for a given seed; OCR makes it take ~20 min on 4 cores):
  `make train-classifier` or `python -m archrender.understand.train --per-class 20`.
  `--cache var/classifier-features` keeps the corpus features (keyed by a hash of the code that
  produces them), so refitting after a change to `understand/classify.py` takes seconds.
  Retrain whenever the feature layout changes: loading refuses a model whose feature list differs.
- **Evaluate** on a held-out corpus: `make eval` (S1 section) or
  `python -m archrender.understand.evaluate`.
- **With the VLM** (UNVERIFIED-ON-GPU): S1 combines this model and the VLM's answer with an
  equal-weight geometric mean and sends confident disagreements to review (DECISIONS.md, Phase-2
  amendments). A combiner calibrated on VLM answers for real pages needs the pod and is not built
  yet.
