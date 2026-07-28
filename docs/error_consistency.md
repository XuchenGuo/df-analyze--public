# Error consistency and repeated K-fold design

## Scientific scope

Error consistency (EC) complements ordinary goodness-of-fit metrics by asking
whether independently refitted models make similar errors on the **same external
holdout rows**. It is a conditional post-selection stability analysis in
df-analyze: preprocessing, selected features, and tuned hyperparameters are fixed
before EC begins.

Classification error IoU implements Equation 1 of
[Levman et al. (2023)](#references). The first four regression definitions were
introduced by [Rahman, Berger, and Levman (2022)](#references). The seven-method
set follows the extended manuscript and reference implementations listed under
[References](#references). The extended manuscript does not provide a DOI or
other public publication identifier, so it is cited as an unpublished
manuscript rather than as a published article.

The regression residual-consistency methods in df-analyze are experimental
descriptive extensions. They are not established inferential statistics,
hypothesis tests, or confidence intervals.

For each of `R` repetitions, the training set is shuffled and partitioned into
`K` folds. One model is fit on the complement of each fold, so there are
`M = K * R` fitted models. Every model predicts the unchanged external holdout.
The main estimate uses all `M * (M - 1) / 2` model pairs.
Ordinary out-of-fold predictions are not substituted because different folds do
not predict the same rows and therefore cannot support samplewise EC directly.

Grouped data use group-aware splitting. If a group-disjoint repeated K-fold
partition cannot be created, that model configuration is skipped rather than
silently falling back to folds with group overlap. Every successful
validation-fold assignment is recorded in `fold_assignments.csv`; a model's
training rows are the complement of its recorded validation fold.

## Classification definition

For model `i`, let `E_i` be the set of external-holdout rows it classifies
incorrectly. Pairwise classification EC is the Jaccard index

```text
EC(i, j) = |E_i intersection E_j| / |E_i union E_j|.
```

Its range is `[0, 1]`, and the optimum is `1`. When both error sets are empty,
the result is mathematically undefined. `--ec-empty-unions` makes that policy
explicit. The default `warn` emits one warning and records the undefined
comparison as `NaN`; finite summary statistics exclude those comparisons. The
`0`, `1`, `nan`, `drop`, and `error` policies remain available for explicit use.

## Regression definitions

Let `r_i(s)` and `r_j(s)` be residuals from two models on holdout sample `s`,
with `a = |r_i(s)|`, `b = |r_j(s)|`, and
`z = sign(r_i(s) * r_j(s))`. The seven implemented definitions are:

| Method | Samplewise value | Optimum | Range | Source |
|---|---:|---:|---:|---|
| `ratio` | `min(a,b) / max(a,b)` | 1 | `[0,1]` | [Rahman et al. (2022)](#references) |
| `ratio_diff` | `|a-b| / (a+b)` | 0 | `[0,1]` | [Rahman et al. (2022)](#references) |
| `ratio_sign` | `z * min(a,b) / max(a,b)` | 1 | `[-1,1]` | [Rahman et al. (2022)](#references) |
| `ratio_diff_sign_magnitude` | primary: `|z * |a-b| / (a+b)|` | 0 | `[0,1]` | [Rahman et al. (2022)](#references), with the aggregation adaptation below |
| `intersection_union_sample` | `I(s) / U(s)` | 1 | `[0,1]` | [extended manuscript](#references) |
| `intersection_union_all` | `sum_s I(s) / sum_s U(s)` | 1 | `[0,1]` | [extended manuscript](#references) |
| `intersection_union_distance` | `D(s)` | 0 | `[0,infinity)` | [extended manuscript](#references) |

For residuals on the same side of zero, `I = min(a,b)`, `U = max(a,b)`, and
`D = |a-b|`. For residuals on opposite sides, `I = 0`, `U = a+b`, and `D = a+b`.
When both residuals are zero, ratio and intersection-over-union values are `1`,
while ratio-difference values and distance are `0`.

For `ratio_diff_sign_magnitude`, the raw signed sample value is retained in
`ec_signed_mean` and `pair_signed_mean` as a direction diagnostic. Its primary
`ec_mean`, pair matrix, and sample profile aggregate the absolute signed value.
This prevents equal positive and negative discrepancies from cancelling to zero
and falsely appearing optimal. The legacy `ratio_diff_sign` spelling remains a
CLI alias and the compatibility method name in existing serialized result
schemas. The supplied manuscript and reference implementation aggregate the
signed values directly, so this magnitude-first summary is an explicit
df-analyze adaptation rather than an exact reproduction of that one summary.

The default `--ec-epsilon 0` follows the unregularized ratio definitions above. For
positive epsilon and two non-zero residual magnitudes, `ratio` and `ratio_sign`
use `(min(a,b) + epsilon) / (max(a,b) + epsilon)`. A zero-over-nonzero endpoint
remains `0`, and two zero residuals remain `1`. The ratio-difference methods add
epsilon to their denominator. This preserves a ratio of 1 for equal residual
magnitudes while keeping the exact zero endpoints. The chosen value is recorded
in the output. Intersection-union distance is
unbounded and scale-dependent; it should be treated as a diagnostic and not
compared numerically across targets with different units.

## Aggregation and dispersion

`ec_mean` retains the all-model-pairs estimate. The output also
separates `within_repetition` and `between_repetition` pairs and reports one mean
within-repetition estimate per repetition. This exposes the dependence structure
without changing the primary estimate.

The dispersion columns have distinct meanings:

- `ec_model_pair_sd`: sample SD of model-pair EC means.
- `ec_pooled_value_sd` (legacy `ec_sd`): SD after pooling pair-by-sample values
  for samplewise regression metrics.
- `ec_sample_profile_sd`: SD across holdout samples after averaging each sample
  over model pairs.
- `EC_scalar_sd` and `EC_vec_sd`: preserved legacy output labels.

All are descriptive dispersions. Model pairs share fitted models and all values
share the same holdout, so these SDs are not standard errors or confidence
intervals. There is no universal dataset-independent EC cutoff.

## Randomness and reproducibility

Split seeds vary deterministically by repetition. `--ec-model-seed-mode vary`
(default) assigns a reproducible, distinct model seed to every repetition/fold,
capturing training-subset and algorithmic instability. `fixed` reuses the base
model seed to isolate training-subset sensitivity as far as the estimator and
hardware permit. Python, NumPy, and PyTorch RNGs are seeded, and recognized
estimator seed arguments are overridden and audited in `trial_design.csv`.

GPU kernels and third-party estimators may still have nondeterministic execution;
the recorded seeds make the experimental intent reproducible but do not promise
bit-for-bit equality on every platform.

The default of five repetitions is a runtime-conscious descriptive default, not a
claim of inferential precision. Increase repetitions when the stability of the
point estimate matters, and report the chosen value. More repetitions do not turn
the dependent model-pair dispersions into confidence intervals.

## Command-line use

Classification uses the error-set intersection-over-union method automatically.
For regression, omitting `--ec-methods` computes all seven methods. A subset can
be selected with, for example:

```shell
--ec \
--ec-folds 5 \
--ec-repetitions 3 \
--ec-methods ratio ratio_diff intersection_union_all
```

The available controls are:

| Option | Default | Meaning |
|---|---:|---|
| `--error-consistency`, `--ec` | off | Enable EC after model tuning and feature selection |
| `--ec-folds` | `5` | Folds per repetition; must be at least 2 |
| `--ec-repetitions` | `5` | Independently shuffled repetitions; must be at least 1 |
| `--ec-model-seed-mode` | `vary` | `vary` includes model-seed variation; `fixed` reuses the base model seed |
| `--ec-methods` | all seven | Regression methods to compute |
| `--ec-save-predictions` | off | Save holdout predictions and residual/error matrices |
| `--ec-empty-unions` | `warn` | Classification policy: `0`, `1`, `nan`, `drop`, `error`, or `warn` |
| `--ec-epsilon` | `0` | Non-negative stabilization for regression ratio denominators |
| `--ec-recurrence-threshold` | `0.5` | Heuristic for the joined adaptive-error/EC report; not a universal cutoff |

See [Command-line arguments](arguments.md) for the generated
CLI help and validation rules.

## Output files

The root `results/error_consistency` directory contains:

- `summary.csv`: EC summaries for every target, model, selection, and method.
- `performance_summary.csv`: predictive performance of the repeated fold models
  on the common holdout.
- `trial_scores.csv`, `trial_design.csv`, and `fold_assignments.csv`: fold-level
  scores, split and seed audit, and exact validation-fold assignments.
- `correlation_summary.csv`, `model_ec_ranking.csv`, and
  `target_ec_trend.csv`: descriptive performance/stability comparisons.
- `metadata.json` and `README.md`: run metadata and interpretation notes.
- `plots/`: EC distributions, EC/performance plots, and correlation heatmaps
  when the required data are available.

Each `<target>/<model>/<selection>_<embed-selector>/` directory contains the
configuration-level trial files, `pairwise_values.csv`,
`pairwise_matrix_<method>.csv`, applicable `sample_ec_<method>.csv` files,
sample and optional group diagnostics, and pairwise plots. Classification
outputs include consistently wrong samples; regression outputs include samples
with large residual consensus. `--ec-save-predictions` additionally writes
`trial_predictions.csv` and `residual_or_error_matrix.csv`.

With multiple external test sets, EC output is nested under `testXX`. When both
adaptive error and EC are enabled, the adaptive-error `tables` directory also
contains `risk_stability_report.csv`, `risk_stability_summary.csv`, and
`risk_stability_skipped.csv`. See [Program outputs](program_outputs.md) for the
project-wide directory layout.

## Interpretation

Rank models by absolute distance from each method's declared optimum, not by an
assumption that larger is always better. EC should accompany, not replace,
predictive performance. When goodness-of-fit is materially different, prefer the
better-performing model; when performance is comparable, EC can be used as a
stability diagnostic or tie-breaker.

## References

1. Levman, J., Ewenson, B., Apaloo, J., Berger, D., and Tyrrell, P. N.
   "Error Consistency for Machine Learning Evaluation and Validation with
   Application to Biomedical Diagnostics." *Diagnostics* 13(7), 1315 (2023).
   [https://doi.org/10.3390/diagnostics13071315](https://doi.org/10.3390/diagnostics13071315)
2. Rahman, M. M., Berger, D., and Levman, J. "Novel Metrics for Evaluation and
   Validation of Regression-based Supervised Learning." *2022 IEEE
   Asia-Pacific Conference on Computer Science and Data Engineering (CSDE)*
   (2022).
   [https://doi.org/10.1109/CSDE56538.2022.10089291](https://doi.org/10.1109/CSDE56538.2022.10089291)
3. Rahman, M. M., Berger, D., Wang, J., Shah, P., Tyrrell, P., and Levman, J.
   "Toward Robust Validation of Regression Models Using Novel Metrics With
   Application to Deep Learning." Unpublished manuscript supplied as a project
   reference; no DOI or public publication identifier was provided.
4. Rahman, M. M. Reference implementations and experiments:
   [simulated datasets](https://github.com/mostafiz67/Regression_EC_Simulation),
   [real-world datasets](https://github.com/mostafiz67/Regression_EC_Real_World_Data),
   [classification versus regression](https://github.com/mostafiz67/Class_EC_VS_Reg_EC),
   [MNIST](https://github.com/mostafiz67/Regression_EC_MNIST),
   [rUNet](https://github.com/mostafiz67/Regression_EC_rUnet), and
   [MIMIC-III](https://github.com/mostafiz67/Regression_EC_MIMIC-III) (2025).
5. Deep Learning Lab. [Classification Error
   Consistency](https://github.com/stfxecutables/error-consistency), reference
   implementation.
