# Calibration and Brier Score

[← Metrics catalog](02-classification-metrics-catalog.md)

A classifier is calibrated when predictions made with probability `p` are correct
about `p` of the time. Among cases assigned roughly 0.8 probability, about 80%
should be positive.

## Discrimination versus calibration

- **Discrimination/ranking:** positives score above negatives.
- **Calibration:** score values correspond to observed frequencies.

A model can rank perfectly but map scores to overconfident probabilities. ROC-AUC
and AP will not reveal that problem.

## Reliability diagram

1. Divide predictions into probability bins.
2. For each bin, compute mean predicted confidence.
3. Compute observed positive frequency.
4. Plot confidence against frequency.

Points on the diagonal are calibrated. Below/above-diagonal interpretations depend
on axis convention, so label axes explicitly. Include bin counts: a sparse bin is
uncertain.

## Expected calibration error

A common ECE definition is:

```text
ECE = sum_b(n_b / N * abs(accuracy_b - confidence_b))
```

ECE is easy to communicate but depends on bin count, bin boundaries, sample size,
and multiclass confidence definition. Report the exact method. ECE can hide
opposing class-specific errors and should accompany a reliability plot.

## Brier score

For binary outcomes:

```text
Brier = mean((p_i - y_i)^2)
```

Lower is better; zero is perfect. The multiclass version sums squared differences
between the predicted probability vector and one-hot target.

Brier score is a proper scoring rule influenced by both calibration and
discrimination. Because prevalence affects its baseline, compare with a predictor
that always emits training/validation prevalence and consider a skill score.

## Calibration methods

- **Temperature scaling:** fit one scalar on validation logits; preserves class
  ranking/argmax but rescales confidence.
- **Platt scaling:** logistic mapping, often for binary scores.
- **Isotonic regression:** flexible monotonic mapping; needs more calibration data.
- **Vector/matrix scaling:** richer multiclass mappings with more overfitting risk.

Fit calibration only on held-out calibration/validation data. Evaluating on the
same samples used to fit the calibrator is optimistic.

## Distribution shift

Calibration is distribution-dependent. Changes in prevalence, population,
sensor, prompt style, or time can invalidate it. Monitor both prediction
distribution and observed outcomes after deployment.

## Recommended report

```text
base and tuned reliability diagrams
Brier score
ECE with binning definition
log loss
class prevalence
sample count per bin
results before and after calibration
untouched test evaluation after fitting calibration on validation data
```
