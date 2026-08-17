# Evaluating and Tuning Performance by Learning Paradigm

[← Readiness plan](../README.md) · [Classification metrics catalog](02-classification-metrics-catalog.md)

“Model performance” has two separate meanings:

1. **Task or statistical performance:** whether the model learned useful behavior.
2. **Systems performance:** how much time, memory, compute, energy, and money are
   required to train or run it.

Every paradigm needs both evaluations, but the correct quality evidence changes
with the training signal. A classifier can be evaluated against labels; a
clustering model may have no ground-truth labels; a reinforcement-learning agent
produces trajectories whose outcomes depend on its actions. Reusing one metric
template across all paradigms creates misleading conclusions.

## Paradigms are not one exclusive list

The terms below answer different questions and can overlap:

| Axis | Question | Examples |
|---|---|---|
| Learning signal | What teaches the model? | supervised, unsupervised, self-supervised, reinforcement, preference learning |
| Label availability | How much human-labeled data exists? | fully supervised, semi-supervised |
| Data acquisition | Which examples receive labels? | passive collection, active learning |
| Update timing | When does learning happen? | batch/offline, online/continual |
| Training location | Where does learning happen? | centralized, federated |

A system can therefore be **active + semi-supervised**, **federated + supervised**,
or **online + reinforcement learning**. Do not force these properties into one
mutually exclusive hierarchy.

## Shared systems-performance ledger

Regardless of paradigm, record:

- hardware, software versions, precision, and power/clock conditions;
- dataset/environment/model revision and random seeds;
- wall-clock training time and accelerator-hours;
- examples, tokens, transitions, or queries processed per second;
- peak device and host memory;
- forward, backward, optimizer, communication, and input-pipeline time;
- inference latency, throughput, batch/concurrency, and model size;
- quality achieved at a fixed resource budget;
- resource cost required to reach a fixed quality target.

The last two measurements are essential. A method that reaches a better final
score using 100 times more labels or environment interactions is not simply
“better”; it occupies a different cost-quality operating point.

## Paradigm comparison

| Paradigm | Primary quality evidence | Main tuning target | Distinct performance bottleneck |
|---|---|---|---|
| Supervised | held-out labeled metrics | generalization from labeled examples | labeled batches, imbalance, output/loss computation |
| Unsupervised | intrinsic structure + downstream/external validation | useful structure without target labels | pairwise distances, large feature matrices, unstable objectives |
| Semi-supervised | held-out labeled metrics vs label budget | exploit unlabeled data without confirmation bias | teacher inference, augmentations, pseudo-label filtering |
| Self-supervised | pretext objective + transfer/probe evaluation | reusable representations or generative modeling | very large data/token volume, negatives/masks, distributed training |
| Reinforcement learning | return/success distributions across episodes | long-term behavior under interaction | environment collection, rollout storage, policy staleness |
| Active learning | quality as a function of labeling budget | select the most valuable labels | repeated acquisition/retraining and pool scoring |
| Preference learning | held-out pairwise preference + downstream behavior | align outputs with comparative judgments | generation cost, pair quality, reward exploitation |

## 1. Supervised learning

### Training signal

Each input `x` has a target `y`. The model minimizes a loss comparing its
prediction with that target.

### Evaluate task performance

Choose metrics from the target type:

- classification: confusion matrix, per-class precision/recall/F1, macro F1,
  balanced accuracy/MCC, PR-AUC, log loss, and calibration;
- regression: MAE, RMSE, residual distributions, quantile/prediction-interval
  coverage, and domain-specific cost;
- ranking: NDCG, MAP, MRR, recall@k, and constraint-based online metrics;
- structured outputs: task-specific exact match, overlap, edit distance, or human
  evaluation.

Use train/validation/test separation. Fit hyperparameters and thresholds on
validation data, then evaluate once on untouched test data. Stratify or group
splits when random splitting would leak identities, time, or near duplicates.

### Tune quality

- learning rate, schedule, optimizer, weight decay, batch size, and epochs;
- architecture/capacity and pretrained initialization;
- class weights, sampling, focal losses, and threshold policy;
- augmentation and regularization;
- full fine-tuning versus frozen layers, LoRA, or other adapters.

### Evaluate and tune systems performance

Measure samples/s, step time, time-to-quality, activation/optimizer memory, data
loader stalls, and inference latency. Optimize data overlap, mixed precision,
compilation, fused operations, batch size, and distributed scaling. Keep target
quality fixed when claiming a speedup.

### Main trap

Optimizing accuracy on an imbalanced dataset can erase minority-class utility.
Report per-class behavior and operational thresholds.

## 2. Unsupervised learning

### Training signal

There is no task label for each example. The objective may reconstruct data,
estimate density, separate clusters, or preserve neighborhood/variance structure.

### Evaluate task performance

No single “unsupervised accuracy” exists. Use several kinds of evidence:

- **intrinsic:** silhouette score, Davies–Bouldin, Calinski–Harabasz,
  reconstruction error, likelihood, or stability;
- **external:** adjusted Rand index, normalized mutual information, or purity only
  when independent labels exist for evaluation—not training;
- **downstream:** linear probe, retrieval quality, anomaly detection, or
  supervised performance using learned representations;
- **stability:** agreement across seeds, resampling, perturbations, or time;
- **human/domain validation:** whether structures are coherent and actionable.

Intrinsic metrics encode assumptions. Silhouette favors compact separated
clusters and can reject valid non-convex structure. A low reconstruction loss can
preserve irrelevant details. Always connect the metric to intended use.

### Tune quality

- cluster count, distance metric, linkage, density radius, and minimum support;
- embedding dimension, regularization, corruption, and reconstruction objective;
- anomaly threshold and contamination assumption;
- preprocessing/scaling, which often determines distance-based results.

Tune using stability and downstream usefulness, not by searching against hidden
test labels and then claiming the process was unsupervised.

### Evaluate and tune systems performance

Measure fit/transform time, memory scaling with samples/features, neighbor or
pairwise-comparison cost, and approximate-index quality/speed. Optimize blocked
linear algebra, sparse representations, mini-batch algorithms, approximate nearest
neighbors, GPU distance kernels, and distributed data partitioning.

### Main trap

Selecting the run that best matches known labels turns those labels into tuning
information and invalidates an unsupervised evaluation claim.

## 3. Semi-supervised learning

### Training signal

A small labeled set is combined with a larger unlabeled set using pseudo-labels,
consistency regularization, entropy objectives, graph propagation, or a
teacher–student model.

### Evaluate task performance

Use an independently labeled test set and report:

- standard supervised task metrics;
- quality versus number of human-labeled examples;
- gain over a supervised-only model trained on exactly the same labeled subset;
- gain over simply labeling more random examples, when cost can be compared;
- pseudo-label precision/coverage and confidence calibration;
- performance as the unlabeled distribution shifts.

The key curve is often **quality versus label budget**, not only final quality.

### Tune quality

- pseudo-label confidence threshold and class balancing;
- unsupervised-loss weight and ramp-up schedule;
- weak/strong augmentation policy;
- teacher update rate and refresh frequency;
- labeled-to-unlabeled batch ratio;
- filtering for distribution mismatch.

### Evaluate and tune systems performance

Teacher inference, multiple augmented views, and larger unlabeled batches can
multiply compute. Report examples/s separately for labeled and unlabeled paths,
teacher/student memory, pseudo-label refresh time, and time-to-quality at a fixed
label budget. Cache teacher outputs only if their staleness is acceptable.

### Main trap

Confirmation bias: incorrect high-confidence pseudo-labels become training targets
and reinforce themselves. Monitor pseudo-label accuracy on a diagnostic labeled
slice without tuning on the final test set.

## 4. Self-supervised learning

### Training signal

Targets are derived from the data itself: masked tokens/patches, next-token
prediction, contrastive views, temporal ordering, or another pretext task.

### Evaluate task performance

Pretraining loss measures the pretext objective, not necessarily representation
usefulness. Combine:

- held-out pretraining loss or perplexity;
- linear-probe performance with frozen representations;
- k-NN/retrieval evaluation;
- few-shot or full downstream fine-tuning;
- transfer across datasets and distribution shifts;
- robustness and representation-collapse diagnostics.

Compare at fixed pretraining tokens/compute and at fixed downstream label budget.

### Tune quality

- masking/corruption rate and objective composition;
- temperature, negative sampling, view augmentation, and batch composition;
- context length, tokenizer, data mixture, and curriculum;
- representation dimension and projection/prediction heads;
- downstream adapter/fine-tuning strategy.

### Evaluate and tune systems performance

Tokens/s or views/s, time-to-loss and time-to-transfer-quality, activation memory,
communication, and data-pipeline cost dominate. Contrastive methods may require
large effective batches or distributed negative exchange. Language-model
pretraining stresses attention, MLP, optimizer state, and parallelism. Optimize
mixed precision, fused kernels, FlashAttention, checkpointing, sharding, and
input/tokenization pipelines.

### Main trap

A lower pretext loss does not prove better downstream representations. Report
transfer evidence and compute/data scale.

## 5. Reinforcement learning

### Training signal

An agent takes actions, changes an environment, and receives rewards. Data is not
independent and identically distributed: the current policy changes which
experiences are collected.

### Evaluate task performance

Report distributions, not one lucky episode:

- mean/median episodic return with confidence intervals;
- success rate, constraint violations, and catastrophic failures;
- sample efficiency: return versus environment steps;
- wall-clock efficiency: return versus elapsed time or accelerator-hours;
- robustness across seeds, environment variants, and initial states;
- regret for appropriate online/bandit settings;
- offline-policy evaluation assumptions when live evaluation is unsafe.

Keep training and evaluation environments/seeds separate. Use deterministic
evaluation policy only when that matches deployment; otherwise evaluate the
actual stochastic policy.

### Tune quality

- learning rates, discount factor, entropy bonus, clipping/trust-region limits;
- rollout length, replay ratio, batch size, target updates, and advantage method;
- reward design/scaling and observation/action normalization;
- exploration schedule and curriculum.

### Evaluate and tune systems performance

Separate environment steps/s from learner updates/s. Profile rollout workers,
simulation, replay buffers, policy inference, learner GPU utilization, and
communication. Tune actor/learner ratios, batching, asynchronous collection,
vectorized environments, replay storage, and policy-update freshness.

### Main traps

- reward hacking: return rises while intended behavior worsens;
- seed sensitivity and selective reporting;
- comparing methods at unequal environment-step or compute budgets;
- policy staleness in asynchronous systems.

## 6. Active learning

### Training signal

Active learning is a **data-acquisition strategy**, usually wrapped around a
supervised or semi-supervised learner. The model selects examples for an oracle
(often a human) to label.

### Evaluate task performance

Plot task quality against:

- number of acquired labels;
- annotation time or monetary cost;
- acquisition rounds;
- total training/inference compute.

Compare against strong baselines: random sampling, stratified sampling, and a
diversity-aware baseline. Reuse identical initial labeled sets and pool/test splits
across strategies. Report variance over several seeds because early acquisitions
can strongly affect later rounds.

### Tune quality

- uncertainty score: entropy, margin, least confidence, ensemble disagreement;
- diversity/representativeness and outlier handling;
- acquisition batch size and round frequency;
- model calibration;
- class-balance or coverage constraints;
- oracle-noise and abstention policy.

### Evaluate and tune systems performance

Pool-wide scoring and embeddings can cost more than training. Measure acquisition
latency, pool examples/s, index memory, retraining time, and human idle time.
Optimize cached embeddings, approximate search, incremental training, batched
acquisition, and asynchronous annotation.

### Main trap

Counting labels while ignoring annotation difficulty. An “informative” example
may take ten times longer to label. Evaluate quality per real annotation cost.

## 7. Preference learning

### Training signal

Humans or synthetic judges compare outputs. The system may train a reward model,
use a direct preference objective such as DPO, or use reinforcement learning
against a learned reward.

### Evaluate task performance

- held-out pairwise preference accuracy and calibration;
- blind win/tie/loss rate against the base model and strong baselines;
- task correctness and safety regression suites;
- judge agreement, positional/order bias, and annotator disagreement;
- KL divergence or behavioral drift from the reference model;
- reward-versus-ground-truth correlation and reward-hacking probes.

Automated judges must be validated against humans and should not evaluate outputs
they generated or strongly resemble.

### Tune quality

- preference-data filtering, balance, and annotator quality;
- DPO/RL temperature or KL coefficient;
- reward-model capacity and regularization;
- response generation temperature and candidate diversity;
- adapter targets, learning rate, and training duration.

### Evaluate and tune systems performance

Generation often dominates cost because each prompt needs multiple candidate
responses. Track tokens generated per preference pair, reward-model throughput,
policy/reference-model memory, and total GPU-hours per accepted improvement.
Optimize shared prefixes/KV caches, batching, quantization of frozen reference or
reward models, and offline generation pipelines.

### Main trap

Improving the learned reward without improving human-valued behavior. Keep
independent human evaluation and safety/capability regression gates.

## Paradigm-specific optimization priorities

| Workload shape | Likely first systems investigation |
|---|---|
| Supervised image/tabular training | input pipeline, batch size, mixed precision, kernel fusion |
| Unsupervised clustering/retrieval | distance computation, memory scaling, approximate search |
| Semi-supervised teacher–student | duplicate forward paths, augmentation, teacher caching |
| Self-supervised Transformer | tokens/s, attention/MLP kernels, activation/optimizer memory, communication |
| Contrastive self-supervision | effective batch, all-gather, negative storage, augmentations |
| Reinforcement learning | environment/learner balance, rollout batching, replay, policy staleness |
| Active learning | pool scoring/indexing, retraining cadence, annotation throughput |
| Preference tuning | generation volume, multi-model memory, KV reuse, judge/reward throughput |

## Common experimental contract

For every paradigm lab, publish:

```text
paradigm and learning signal
task-quality metric and why it is valid
systems metric and workload unit
data / environment / oracle budget
baseline
quality-versus-resource curve
hyperparameters and selection protocol
hardware and software environment
raw measurements and variability
failure cases and distribution shifts
```

## Recommended paradigm labs

1. **Supervised:** fine-tune a classifier; optimize macro F1 under a latency limit.
2. **Unsupervised:** cluster embeddings; compare intrinsic scores, stability, and
   downstream linear-probe value.
3. **Semi-supervised:** vary labeled fraction and pseudo-label threshold; plot
   quality against labels and GPU-hours.
4. **Self-supervised:** pretrain a small masked or contrastive model; compare
   pretext loss with transfer quality and throughput.
5. **Reinforcement:** train a vectorized-environment agent; plot return against
   environment steps and wall time while profiling actor/learner balance.
6. **Active:** compare uncertainty and random acquisition using identical label
   budgets and annotation-cost assumptions.
7. **Preference:** perform a small DPO experiment; report blind win rate, KL drift,
   regressions, generation cost, and training throughput.

## Final decision rule

Do not ask only, “Which model has the highest score?” Ask:

```text
Which method achieves the required task behavior
under the available labels/data/interactions,
compute and memory budget,
latency/throughput constraint,
and acceptable failure risk?
```

That complete operating point—not one isolated metric—is model performance.
