# Ranking Metrics: PR-AUC, Average Precision, and ROC-AUC

[← Metrics catalog](02-classification-metrics-catalog.md)

Ranking metrics evaluate whether positive examples receive higher scores than
negative examples across thresholds. They do not prove that probabilities are
calibrated or that one deployment threshold is acceptable.

## Precision-recall curves and average precision

A precision-recall (PR) curve evaluates every score threshold:

- lowering the threshold usually increases recall and reduces precision;
- raising it usually increases precision and reduces recall.

**Average precision (AP)** summarizes precision at successive recall changes:

```text
AP = sum_n((recall_n - recall_(n-1)) * precision_n)
```

Implementations differ in interpolation details, so record the library/version.
PR-AUC is sometimes used loosely for AP and sometimes for trapezoidal integration;
do not assume they are identical.

The no-skill precision baseline is approximately the positive prevalence. An AP
of 0.20 is excellent if prevalence is 0.001, but poor if prevalence is 0.50.
Always report prevalence.

## ROC curve and ROC-AUC

The ROC curve plots:

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

ROC-AUC can be interpreted as the probability that a randomly chosen positive
receives a higher score than a randomly chosen negative (with tie handling).

ROC-AUC is useful for overall rank discrimination, but severe imbalance can make
small FPR values hide a large absolute number of false positives. PR curves focus
directly on positive predictions and are generally more revealing for rare-event
detection.

## Multiclass and multilabel ranking

Compute one-versus-rest curves per class, then state whether the summary is macro,
weighted, or micro. Macro AP weights rare and common classes equally. Micro AP
pools label decisions and can be dominated by frequent labels.

## Partial and constrained metrics

Full-curve area may include operating regions that are unusable. Prefer metrics
aligned to constraints when possible:

- precision at 90% recall;
- recall at 99% precision;
- partial ROC-AUC for `FPR <= 0.01`;
- precision@k for a review budget;
- recall within the first-stage candidate budget.

## Correct evaluation protocol

1. Produce scores on validation data.
2. Compare curves and choose an operational constraint.
3. Select the threshold on validation data.
4. Lock it.
5. Report test-set decision metrics at that threshold.
6. Include AP/PR-AUC as ranking context, not as the deployed operating point.

## Common mistakes

- Reporting ROC-AUC alone for a rare positive class.
- Comparing AP values without comparing prevalence.
- Calling trapezoidal PR-AUC and average precision interchangeable.
- Treating high AUC as calibrated probabilities.
- Selecting the deployment threshold on test data.
- Hiding per-class failures behind a micro average.
