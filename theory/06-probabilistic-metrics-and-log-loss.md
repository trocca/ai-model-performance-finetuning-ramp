# Probabilistic Metrics and Log Loss

[← Metrics catalog](02-classification-metrics-catalog.md)

Decision metrics discard score magnitude. Log loss evaluates the probability
assigned to the observed outcome and strongly penalizes confident mistakes.

## Binary log loss

For target `y in {0,1}` and predicted probability `p`:

```text
loss = -(y * log(p) + (1-y) * log(1-p))
```

The dataset log loss is the mean across examples. For a positive example:

```text
p=0.90 -> loss=0.105
p=0.51 -> loss=0.673
p=0.01 -> loss=4.605
```

All but the last may yield the same class decision under a 0.5 threshold, yet log
loss rewards the confident correct prediction and heavily penalizes the confident
wrong one.

## Multiclass cross-entropy

For one-hot target class `y` and predicted class distribution `p`:

```text
loss = -log(p_y)
```

Only the probability assigned to the correct class enters this expression, but
softmax normalization couples all class logits.

## Compute from logits

Use numerically stable library operations such as PyTorch
`torch.nn.functional.binary_cross_entropy_with_logits` and
`torch.nn.functional.cross_entropy`. Do not manually apply softmax/sigmoid and
then take logs during training; extreme values can underflow.

## Interpretation

Lower is better, and zero is perfect. The scale depends on task uncertainty and
number of classes, so compare against:

- the untouched base model;
- a prevalence-only or uniform predictor;
- the same dataset and preprocessing;
- confidence intervals or repeated runs.

## Log loss versus accuracy

Accuracy measures final correctness. Log loss measures probability quality under
a logarithmic scoring rule. A fine-tuned model can keep identical accuracy while
improving log loss, or improve accuracy while becoming dangerously overconfident.
Report both when downstream systems consume probabilities.

## Caveats

- Label noise makes confident fitting expensive under log loss.
- Class weighting changes the optimized objective and its interpretation.
- Temperature scaling can improve log loss/calibration without changing ranking
  or argmax accuracy.
- Never compare mean losses produced with different masking or reduction rules.
- In language modeling, state whether loss is per token, which tokens are masked,
  and whether perplexity is `exp(mean token loss)`.

## Fine-tuning checklist

- [ ] Use stable loss-from-logits functions.
- [ ] Record class weights, label smoothing, masks, and reduction.
- [ ] Compare base and tuned models on identical examples.
- [ ] Report accuracy/F1 beside log loss.
- [ ] Inspect calibration after fine-tuning.
- [ ] Avoid choosing checkpoints using the final test loss.
