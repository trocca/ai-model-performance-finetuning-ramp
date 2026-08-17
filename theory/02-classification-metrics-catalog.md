# Classification Metrics Catalog

[← Readiness plan](../README.md) · [Accuracy, precision, and recall](01-accuracy-precision-recall.md)

Classification metrics evaluate different objects: final decisions, ranked
scores, probabilities, label sets, agreement, or operational cost. This catalog
is sorted by **default practical relevance**, not by mathematical importance.
Change the order when the task's risks demand it.

## Recommended minimum report

For most imbalanced classification and fine-tuning evaluations, report all of:

1. [Confusion matrix with absolute counts](01-accuracy-precision-recall.md#start-with-the-positive-class).
2. [Class prevalence and per-class support](03-multiclass-averaging-and-support.md#prevalence-and-support).
3. [Per-class precision, recall, and F1](03-multiclass-averaging-and-support.md#per-class-metrics).
4. [Macro F1 and micro F1](03-multiclass-averaging-and-support.md#macro-micro-and-weighted-averages).
5. [Balanced accuracy or MCC](04-imbalance-aware-summary-metrics.md).
6. [PR-AUC / average precision](05-ranking-metrics-and-curves.md#precision-recall-curves-and-average-precision).
7. [Log loss / cross-entropy](06-probabilistic-metrics-and-log-loss.md#log-loss).
8. [Calibration plot and Brier score](07-calibration-and-brier-score.md).
9. [Metrics at the deployment threshold](08-threshold-selection-and-operating-points.md).

For balanced problems with comparable error costs, start with accuracy,
per-class precision/recall/F1, and the confusion matrix, then add probability and
calibration metrics when scores are consumed as probabilities.

## Ranked catalog

| Rank | Metric | What it answers | Most relevant when |
|---:|---|---|---|
| 1 | Confusion matrix | Which error types occurred, and how often? | Always; raw counts anchor every derived rate. |
| 2 | Precision | How trustworthy are positive predictions? | False positives are costly. |
| 3 | Recall / sensitivity / TPR | How many real positives were found? | False negatives are costly. |
| 4 | Accuracy | What fraction of all decisions were correct? | Classes and error costs are balanced. |
| 5 | F1 | Are precision and recall jointly strong? | Both positive-class error types matter. |
| 6 | Per-class precision/recall/F1 | Which classes work or fail? | Multiclass and multilabel tasks. |
| 7 | PR-AUC / average precision | Does ranking preserve rare positives across thresholds? | Positive class is rare. |
| 8 | Specificity / TNR | How many actual negatives were rejected? | Negative-class protection matters. |
| 9 | False-positive rate | What fraction of negatives trigger false alarms? | Large negative populations and alerting. |
| 10 | ROC-AUC | How well are positives ranked above negatives? | Threshold-independent comparison with moderate balance. |
| 11 | Balanced accuracy | Are positive and negative recall jointly strong? | Binary or multiclass imbalance. |
| 12 | Matthews correlation coefficient | How correlated are predictions and truth using all cells? | A robust single-number binary summary is needed. |
| 13 | Log loss / cross-entropy | How much probability did the model assign to the truth? | Probability quality and training evaluation. |
| 14 | Calibration error / reliability diagram | Do stated probabilities match observed frequencies? | Decisions depend on risk estimates. |
| 15 | Brier score | How close are predicted probabilities to outcomes? | Calibration and discrimination both matter. |
| 16 | Top-k accuracy | Does the correct class appear among the top candidates? | Many-class recognition or candidate generation. |
| 17 | F-beta | What precision/recall balance reflects asymmetric risk? | Recall or precision deserves explicit extra weight. |
| 18 | Negative predictive value | How trustworthy are negative predictions? | Negative decisions release or clear cases. |
| 19 | False-negative rate | What fraction of positives are missed? | Miss rate is the operational language. |
| 20 | Cohen's kappa | How much agreement exceeds chance agreement? | Annotator/model agreement studies. |
| 21 | Jaccard / IoU | How much do predicted and true positive sets overlap? | Multilabel tasks and segmentation. |
| 22 | Hamming loss | What fraction of individual label decisions are wrong? | Multilabel classification. |
| 23 | Subset accuracy / exact match | How often is the entire label set exactly correct? | Complete multilabel outputs must match. |
| 24 | Precision@k / recall@k / hit rate@k | Is relevant content in the first `k` results? | Retrieval and recommendation. |
| 25 | Mean average precision (mAP) | How good is ranked retrieval across queries/classes? | Retrieval, detection, and multilabel ranking. |
| 26 | Coverage error / ranking loss | How deeply must one search, or how often is label order wrong? | Multilabel ranking. |
| 27 | Expected cost / cost-weighted error | What is the actual consequence of mistakes? | Error costs can be quantified. |
| 28 | Lift and cumulative gain | How much better is targeting than random selection? | Marketing, fraud review, and risk queues. |
| 29 | Diagnostic odds ratio | How strongly does a test separate conditions? | Medical diagnostic summaries. |
| 30 | Positive/negative likelihood ratios | How does a result update prior odds? | Clinical and Bayesian diagnostic reasoning. |

## Decision metrics

These require a threshold or final class choice:

- confusion matrix, accuracy, precision, recall, specificity, FPR, FNR, NPV;
- F1/F-beta, balanced accuracy, MCC, kappa, Jaccard;
- Hamming loss and subset accuracy.

They describe one operating point. Always record the threshold and decision rule.

## Ranking metrics

These evaluate score order over many thresholds or at a result cutoff:

- PR-AUC / average precision and ROC-AUC;
- precision@k, recall@k, hit rate@k, and mAP;
- coverage error, ranking loss, lift, and gain.

Ranking can be excellent while probabilities are badly calibrated.

## Probability metrics

These evaluate the full predictive distribution:

- log loss / cross-entropy;
- Brier score;
- expected/max calibration error and reliability diagrams.

Probability metrics distinguish a correct 0.51 prediction from a correct 0.99
prediction, unlike accuracy.

## Operational metrics

Expected cost, constraint-based precision/recall, alert volume, and capacity at
`k` connect model behavior to the real system. They are often more actionable
than maximizing a generic score.

## Reporting rule

Never publish one isolated number. State:

- task type and positive class;
- dataset and split;
- class support/prevalence;
- threshold and averaging mode;
- confidence interval or seed variation where appropriate;
- confusion counts;
- comparison baseline;
- operational constraint and failure cost.
