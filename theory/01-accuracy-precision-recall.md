# Accuracy, Precision, and Recall

[← Readiness plan](../README.md)

Accuracy, precision, and recall evaluate **classification decisions**. They answer
different questions, so none is universally “the best metric.” The appropriate
metric depends on class balance and on the real cost of false positives and false
negatives.

> In this chapter, **precision** is a classification metric. It is unrelated to
> numerical precision such as FP32, BF16, FP16, or INT8.

## Start with the positive class

For binary classification, choose one class as **positive**. If a model detects
fraud, “fraud” is normally positive and “legitimate transaction” is negative.
The names do not mean morally good or bad; they identify the event being detected.

Every prediction falls into one cell of a confusion matrix:

| | Actually positive | Actually negative |
|---|---:|---:|
| **Predicted positive** | True positive (`TP`) | False positive (`FP`) |
| **Predicted negative** | False negative (`FN`) | True negative (`TN`) |

- **True positive:** the model correctly detects a positive case.
- **False positive:** the model raises a positive prediction for a negative case.
- **False negative:** the model misses a positive case.
- **True negative:** the model correctly rejects a negative case.

All three metrics below are ratios constructed from these four counts.

## Accuracy

Accuracy asks:

> Of all predictions, what fraction were correct?

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy includes correct decisions from both classes. It is easy to communicate
and useful when:

- classes are reasonably balanced;
- false positives and false negatives have similar costs;
- each example contributes equally;
- one overall fraction correctly summarizes the task.

### The class-imbalance trap

Suppose 990 of 1,000 transactions are legitimate and 10 are fraudulent. A model
that always predicts “legitimate” has:

```text
TN = 990
FN = 10
TP = 0
FP = 0

accuracy = 990 / 1000 = 99%
```

The reported accuracy looks excellent, but the detector catches no fraud. High
accuracy can therefore hide complete failure on a rare class.

### Accuracy is not model confidence

Accuracy uses final class decisions. It does not say whether the winning score
was 0.51 or 0.99, whether probabilities are calibrated, or how performance changes
under a different threshold.

## Precision

Precision asks:

> When the model predicts positive, how often is it correct?

```text
precision = TP / (TP + FP)
```

The denominator is **all predicted positives**. Precision becomes important when
false positives are costly.

Examples:

- Messages sent to spam should rarely be legitimate.
- Automatic account suspension should rarely punish an innocent user.
- An expensive manual investigation should receive mostly useful alerts.
- A retrieved-document set should contain mostly relevant documents.

High precision means positive predictions are trustworthy. It does not guarantee
that the model finds most positive cases. A model can obtain very high precision
by predicting positive only for a few extremely obvious examples.

### Precision vocabulary

Precision is also called **positive predictive value** (`PPV`). In information
retrieval, it is the fraction of retrieved items that are relevant.

## Recall

Recall asks:

> Of all actual positive cases, what fraction did the model find?

```text
recall = TP / (TP + FN)
```

The denominator is **all actual positives**. Recall becomes important when false
negatives are costly.

Examples:

- A cancer-screening system should miss as few cancers as possible.
- A safety monitor should catch dangerous events.
- A fraud screen should find fraudulent transactions for later review.
- A search system's candidate stage should retain relevant documents for ranking.

High recall means few positives are missed. It does not guarantee that positive
predictions are trustworthy. Predicting every example as positive achieves 100%
recall but may produce terrible precision.

### Recall vocabulary

Recall is also called **sensitivity** or the **true-positive rate** (`TPR`).

## One worked example

Consider 1,000 cases:

```text
TP = 40   correctly detected positives
FP = 10   false alarms
FN = 20   missed positives
TN = 930  correctly rejected negatives
```

Then:

```text
accuracy  = (40 + 930) / 1000 = 0.970 = 97.0%
precision = 40 / (40 + 10)    = 0.800 = 80.0%
recall    = 40 / (40 + 20)    = 0.667 = 66.7%
```

All three statements are simultaneously true:

- the model gets 97% of all cases right;
- 80% of its positive alerts are correct;
- it finds about two thirds of the actual positive cases.

The same model can therefore look strong under accuracy and still miss one third
of the positive class.

## Precision and recall trade off through the threshold

Many binary classifiers output a score or probability rather than a class. A
threshold converts the score into a decision:

```text
predict positive if score >= threshold
```

The usual behavior is:

| Threshold change | Predicted positives | Typical precision | Typical recall |
|---|---:|---:|---:|
| Raise threshold | fewer | increases | decreases |
| Lower threshold | more | decreases | increases |

This is a tendency, not a mathematical guarantee at every individual threshold.
The correct threshold is a product decision based on error costs, capacity, and
required service levels—not automatically `0.5`.

Examples:

- A first-stage medical screen may use a low threshold for high recall, followed
  by a more precise test.
- A fully automatic destructive action may require a high threshold for high
  precision.
- An alerting system may choose the highest recall possible while limiting false
  alerts to what investigators can process.

## F1 score

F1 combines precision and recall using their harmonic mean:

```text
F1 = 2 * precision * recall / (precision + recall)
   = 2TP / (2TP + FP + FN)
```

For the worked example:

```text
F1 = 2 * 0.80 * 0.667 / (0.80 + 0.667) = 0.727 = 72.7%
```

The harmonic mean is pulled down by the smaller value, so a model needs both good
precision and good recall for a high F1 score. F1 is useful when:

- the positive class is important;
- class imbalance makes accuracy misleading;
- false positives and false negatives should receive roughly equal emphasis.

F1 ignores true negatives. It can therefore be inappropriate when correct
negative decisions or the false-positive rate are operationally important.

The generalized `F_beta` score weights recall more when `beta > 1` and precision
more when `beta < 1`:

```text
F_beta = (1 + beta^2) * precision * recall
         / (beta^2 * precision + recall)
```

## Specificity and false-positive rate

Recall measures coverage of positives. Its negative-class counterpart is
**specificity**:

```text
specificity = TN / (TN + FP)
```

Specificity asks what fraction of actual negatives were correctly rejected. The
false-positive rate is:

```text
FPR = FP / (FP + TN) = 1 - specificity
```

When negatives are extremely numerous, even a small false-positive rate can
create a large number of false alerts. This is why precision should be examined
together with prevalence and absolute confusion-matrix counts.

## Precision-recall and ROC curves

Metrics at one threshold describe only one operating point.

- A **precision-recall curve** plots precision against recall over thresholds. It
  is usually more informative when the positive class is rare.
- A **receiver operating characteristic (ROC) curve** plots recall/TPR against
  FPR. ROC-AUC measures ranking across thresholds, but can look optimistic on
  highly imbalanced data because the many true negatives keep FPR small.
- **Average precision** or PR-AUC summarizes the precision-recall curve. Its
  baseline depends on positive-class prevalence, so report prevalence too.

Do not choose a deployment threshold by maximizing an area-under-curve metric.
Choose it on validation data using the operational cost or constraint, then report
performance once on untouched test data.

## Multiclass classification

For `K` mutually exclusive classes, treat each class in turn as positive and all
other classes as negative. This produces per-class precision and recall.

How those values are averaged matters:

- **Macro average:** compute the metric per class, then average equally. Rare
  classes matter as much as common classes.
- **Weighted macro average:** average per-class metrics weighted by class support.
  Common classes dominate more.
- **Micro average:** sum TP, FP, and FN across classes, then compute the metric.
  Each individual example contributes equally.

For ordinary single-label multiclass classification, micro precision, micro
recall, and accuracy are equal because every wrong prediction creates one false
positive and one false negative. Macro scores can still reveal weak rare classes.

Always report the averaging rule. “F1 = 0.91” is incomplete without saying
whether it is micro, macro, weighted, or per-class.

## Multilabel classification

In multilabel classification, one example can have several positive labels.
Thresholds can be global or label-specific. Useful summaries include:

- micro and macro precision/recall/F1;
- per-label metrics and label support;
- subset accuracy, which requires the entire predicted label set to match;
- sample-averaged metrics, computed per example and then averaged.

Subset accuracy is strict and can be low even when most individual label decisions
are correct.

## Undefined edge cases

If the model predicts no positives, `TP + FP = 0`, so precision has a zero
denominator. If the test set contains no actual positives, `TP + FN = 0`, so
recall is undefined.

Libraries may return zero, one, `NaN`, or a warning depending on configuration.
Do not hide the case with arbitrary arithmetic. Report that the metric is
undefined and inspect the data split or model behavior.

## PyTorch implementation from counts

This example keeps metric computation explicit:

```python
import torch


def binary_metrics(logits, targets, threshold=0.5):
    """Return confusion counts and metrics for targets containing 0 or 1."""
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    positives = targets.bool()

    tp = (predictions & positives).sum()
    fp = (predictions & ~positives).sum()
    fn = (~predictions & positives).sum()
    tn = (~predictions & ~positives).sum()

    total = tp + fp + fn + tn
    predicted_positive = tp + fp
    actual_positive = tp + fn

    accuracy = (tp + tn).float() / total
    precision = (
        tp.float() / predicted_positive
        if predicted_positive > 0
        else torch.tensor(float("nan"))
    )
    recall = (
        tp.float() / actual_positive
        if actual_positive > 0
        else torch.tensor(float("nan"))
    )

    return {
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tn": tn.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }
```

For production evaluation, use a tested metrics library such as TorchMetrics or
scikit-learn, but still understand its task mode, threshold, averaging, ignored
labels, distributed synchronization, and zero-division policy.

## Fine-tuning evaluation guidance

For a base model versus fine-tuned model comparison:

1. Freeze the task definition and positive class before training.
2. Keep train, validation, and test splits separate.
3. Tune hyperparameters and thresholds on validation data only.
4. Report the test confusion matrix and absolute counts.
5. Report class prevalence and per-class support.
6. Include accuracy, precision, and recall only when each answers a relevant
   product or research question.
7. Add F1, PR-AUC, calibration, or task-specific metrics where appropriate.
8. Compare at a shared threshold and at shared operational constraints—for
   example, precision at 90% recall.
9. Add confidence intervals or repeated-seed variation when the test set is small.
10. Evaluate important data slices; a global average can hide subgroup failure.

For language-model generation, these metrics apply only when the task can be
defined as classification or retrieval. Open-ended generation generally needs
different automated and human evaluation methods.

## Choosing the metric

| Situation | Primary emphasis | Reason |
|---|---|---|
| Balanced classes and similar error costs | Accuracy | Overall correctness is representative. |
| False positives are expensive | Precision | Positive predictions must be trustworthy. |
| False negatives are expensive | Recall | Actual positives must be found. |
| Rare positive class, both error types matter | Precision + recall + F1/PR curve | Accuracy can hide minority failure. |
| Fixed investigation capacity | Precision/recall at top `K` or alert budget | The system can act on only a limited number. |
| Safety screen followed by expert review | Recall under an acceptable false-positive burden | Missing a case is worse than reviewing extra cases. |
| Automatic high-impact action | Precision under a required recall floor | False action must be rare without ignoring coverage. |

## Common mistakes

- Reporting accuracy alone on imbalanced data.
- Forgetting to define which class is positive.
- Confusing precision with floating-point precision.
- Saying high precision means the model finds most positives; that is recall.
- Saying high recall means alerts are usually correct; that is precision.
- Comparing models at different thresholds without stating them.
- Selecting a threshold on the test set.
- Reporting percentages without confusion-matrix counts or class prevalence.
- Mixing macro, micro, and weighted averages.
- Treating F1 as universally superior to accuracy.
- Ignoring probability calibration because ranking metrics look good.

## Mastery check

Given:

```text
TP = 72, FP = 18, FN = 8, TN = 902
```

Calculate:

1. accuracy;
2. precision;
3. recall;
4. F1;
5. specificity;
6. what should happen to precision and recall if the decision threshold is raised.

Expected values:

```text
accuracy    = (72 + 902) / 1000 = 97.4%
precision   = 72 / 90           = 80.0%
recall      = 72 / 80           = 90.0%
F1          = 2 * .8 * .9 / 1.7 = 84.7%
specificity = 902 / 920         = 98.0%
```

Raising the threshold will typically increase precision and decrease recall.
