# Threshold Selection and Deployment Operating Points

[← Metrics catalog](02-classification-metrics-catalog.md)

A model score becomes an action only after a decision policy. Threshold selection
is therefore part of system design, not a cosmetic evaluation detail.

## Why 0.5 is not automatically correct

A 0.5 threshold assumes a particular probability interpretation and cost
relationship. Class weighting, resampling, imperfect calibration, unequal error
costs, and limited review capacity all invalidate blind reliance on 0.5.

## Constraint-based selection

Choose a rule that matches the operation:

- maximize recall subject to precision `>= 0.95`;
- maximize precision subject to recall `>= 0.90`;
- keep false-positive rate below `0.1%`;
- return the top 500 alerts per day;
- minimize expected monetary/clinical cost;
- maximize F-beta when its weighting is justified.

## Expected-cost thresholding

If costs can be estimated:

```text
expected_cost = FP * cost_FP + FN * cost_FN
```

Add review, delay, intervention, and missed-opportunity costs where relevant.
Costs may depend on subgroups or severity, requiring more than one global rule.

## Capacity-limited systems

When humans can review only `K` cases, rank by score and report precision@k,
recall@k, and lift over random selection. A fixed threshold can generate unstable
volume when prevalence drifts; a top-k policy stabilizes volume but changes its
score cutoff over time.

## Validation protocol

1. Define positive class and operational constraint before test evaluation.
2. Train the model on training data.
3. Generate scores on validation data.
4. Fit any probability calibrator on held-out validation/calibration data.
5. Choose the threshold or top-k policy on validation data.
6. Freeze model, calibrator, preprocessing, and policy.
7. Evaluate exactly once on untouched test data.
8. Report confidence intervals and important slices.

## Deployment report

Include:

| Item | Required detail |
|---|---|
| Policy | threshold, top-k, or cost rule |
| Score | logit, probability, calibrated probability, or rank score |
| Validation criterion | exact constraint/objective used to select policy |
| Test counts | TP, FP, FN, TN and support |
| Test metrics | precision, recall, F1, specificity, expected volume/cost |
| Uncertainty | bootstrap interval or repeated-window variation |
| Slices | high-risk groups, sources, time windows, and rare classes |
| Monitoring | prevalence, score drift, alert volume, delayed labels, calibration |

## Thresholds under drift

If prevalence rises, precision often changes even when sensitivity and
specificity remain stable. Monitor actual alert counts and delayed ground truth.
Thresholds should be reviewed through a controlled process, not silently tuned on
production outcomes without a new holdout evaluation.

## Comparing base and fine-tuned models

Compare both:

1. at the same fixed threshold, to expose score-distribution changes;
2. at the same operational constraint, such as 90% recall, to compare achievable
   precision fairly;
3. across the full PR curve, to compare ranking.

Never claim the tuned model is better merely because it uses a more favorable
threshold.
