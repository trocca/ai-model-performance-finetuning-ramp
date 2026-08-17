# Multiclass Averaging, Prevalence, and Support

[← Metrics catalog](02-classification-metrics-catalog.md)

One overall score can conceal a failed class. Multiclass and multilabel reports
therefore need per-class results and an explicit averaging rule.

## Prevalence and support

**Support** is the number of actual examples belonging to a class. **Prevalence**
is its fraction of the evaluated population:

```text
support_c = number of examples whose true class is c
prevalence_c = support_c / total examples
```

Report both because percentages computed from 10 examples are less stable than
percentages computed from 10,000. Deployment prevalence may also differ from the
test set, changing precision even when class-conditional behavior is unchanged.

## One-versus-rest counts

For each class `c`, treat `c` as positive and every other class as negative. This
produces `TP_c`, `FP_c`, `FN_c`, and `TN_c`, then:

```text
precision_c = TP_c / (TP_c + FP_c)
recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
```

## Per-class metrics

Per-class precision identifies classes polluted by predictions from other
classes. Per-class recall identifies classes the model fails to find. Inspect the
full confusion matrix to learn which pairs are confused.

A minimum table contains:

| Class | Support | Prevalence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| A | ... | ... | ... | ... | ... |

## Macro, micro, and weighted averages

### Macro average

```text
macro_F1 = (F1_1 + F1_2 + ... + F1_K) / K
```

Every class receives equal weight. Macro metrics expose rare-class weakness and
are usually the most informative summary when all classes matter.

### Weighted macro average

```text
weighted_F1 = sum_c(support_c * F1_c) / total_support
```

Large classes dominate. Weighted F1 can look strong while a rare class fails, so
keep the per-class table beside it.

### Micro average

First pool counts, then calculate:

```text
micro_precision = sum(TP_c) / (sum(TP_c) + sum(FP_c))
micro_recall    = sum(TP_c) / (sum(TP_c) + sum(FN_c))
```

Each individual label decision contributes equally. In ordinary single-label
multiclass classification, micro precision, micro recall, micro F1, and accuracy
are equal. This equality does not generally hold for multilabel tasks.

## Worked example

Suppose class supports are `A=900`, `B=90`, `C=10`, with F1 scores `0.98`, `0.75`,
and `0.10`:

```text
macro F1    = (0.98 + 0.75 + 0.10) / 3 = 0.61
weighted F1 = (900*.98 + 90*.75 + 10*.10) / 1000 = 0.951
```

The 0.951 weighted score describes the majority experience; the 0.61 macro score
warns that performance is not equitable across classes. Neither should replace
the per-class row showing class C at 0.10.

## Multilabel choices

One sample can own multiple labels, so also consider:

- sample-averaged precision/recall/F1;
- per-label thresholds;
- Hamming loss for individual label decisions;
- subset accuracy for exact label-set matches;
- label cardinality and label co-occurrence slices.

## Fine-tuning report checklist

- [ ] Per-class support and prevalence.
- [ ] Confusion matrix.
- [ ] Per-class precision, recall, and F1.
- [ ] Macro and micro F1.
- [ ] Weighted F1 only as an additional summary.
- [ ] Zero-division behavior stated.
- [ ] Important slices evaluated separately.
- [ ] Base and tuned models evaluated on the identical examples.
