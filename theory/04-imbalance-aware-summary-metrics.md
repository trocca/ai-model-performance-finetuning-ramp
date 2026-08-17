# Balanced Accuracy and Matthews Correlation Coefficient

[← Metrics catalog](02-classification-metrics-catalog.md)

Accuracy can be dominated by a majority class. Balanced accuracy and Matthews
correlation coefficient (MCC) provide stronger summaries under imbalance, but
they answer different questions.

## Balanced accuracy

For binary classification:

```text
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
balanced_accuracy = (sensitivity + specificity) / 2
```

It gives the positive and negative classes equal weight regardless of prevalence.
For multiclass problems, it is the macro average of per-class recall.

An always-negative classifier on a dataset with 99% negatives has 99% ordinary
accuracy but:

```text
sensitivity = 0
specificity = 1
balanced_accuracy = 0.5
```

The 0.5 score correctly exposes chance-like class-balanced behavior.

## Matthews correlation coefficient

Binary MCC uses all four confusion-matrix cells:

```text
MCC = (TP*TN - FP*FN)
      / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Interpretation:

- `+1`: perfect agreement;
- `0`: no correlation between prediction and truth;
- `-1`: perfect inverse prediction.

MCC is demanding: a high score generally requires good behavior on both classes.
It remains informative when class sizes differ sharply and has multiclass
generalizations.

## Which should be primary?

Use **balanced accuracy** when stakeholders understand recall and want equal
average recall across classes. Use **MCC** when a robust single-number correlation
summary is valuable. Report confusion counts and per-class metrics beside either;
no scalar reveals the error pattern.

## Caveats

- Balanced accuracy treats classes equally even if their real costs are unequal.
- MCC can be undefined when denominator terms are zero; state library behavior.
- Neither evaluates probability calibration or ranking across thresholds.
- Both depend on a selected threshold.
- Resampling the test set to artificial balance changes prevalence-dependent
  metrics such as precision; prefer evaluating on realistic prevalence too.

## Recommended report

```text
threshold:
class prevalence:
TP / FP / FN / TN:
ordinary accuracy:
balanced accuracy:
MCC:
per-class precision / recall / F1:
```
