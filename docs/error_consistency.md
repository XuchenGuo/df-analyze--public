# Error consistency

Error consistency (EC) asks a simple question: when the same model is trained
again, does it make similar errors on the same samples?

Accuracy, MAE, and other performance scores tell us how much error a model
makes. EC tells us whether repeated fits fail in the same places. Two models can
have the same accuracy or MAE and still have very different EC.

## How df-analyze runs EC

`df-analyze` runs EC after preprocessing, feature selection, and hyperparameter
tuning:

1. It keeps the selected features and tuned parameters fixed.
2. It splits the training data into `K` folds.
3. It fits one model on the complement of each fold.
4. It repeats the shuffled K-fold split `R` times.
5. Every fitted model predicts the same external holdout rows.
6. It compares every pair of fitted models.

This produces `K * R` models for each target, model, and selected feature set.
The number of model pairs is:

```text
(K * R) * (K * R - 1) / 2
```

The repeated folds change the rows used for fitting. They do not change the
holdout rows used to calculate EC. This is different from ordinary repeated
K-fold cross-validation, where each fold is evaluated on a different set of
rows.

For grouped data, df-analyze keeps groups in separate folds. If it cannot make a
valid group-separated split, it skips that configuration and records the
reason. It does not silently fall back to overlapping groups.

## Quick start

Run these commands from the repository root. Both examples use 2 folds and
2 repetitions, so each configuration is refitted 4 times.

Classification:

```shell
python df-analyze.py \
    --df data/small_classifier_data.json \
    --target target \
    --mode classify \
    --classifiers lr dummy \
    --feat-select filter \
    --htune-trials 1 \
    --ec \
    --ec-folds 2 \
    --ec-repetitions 2 \
    --ec-output-detail summary \
    --outdir ./ec_classification_results
```

Regression:

```shell
python df-analyze.py \
    --df data/testing/regression/forest_fires/forest_fires.parquet \
    --target target \
    --mode regress \
    --regressors elastic dummy \
    --feat-select filter \
    --htune-trials 1 \
    --ec \
    --ec-folds 2 \
    --ec-repetitions 2 \
    --ec-methods ratio ratio_diff intersection_union_all \
    --ec-output-detail summary \
    --outdir ./ec_regression_results
```

The regression command calculates three methods to keep the first run short.
Omit `--ec-methods` to calculate all seven default methods.

## Classification EC

For model `i`, let `E_i` be the holdout rows it classifies incorrectly. EC for
models `i` and `j` is the intersection-over-union of their error sets:

```text
EC(i, j) = |E_i intersection E_j| / |E_i union E_j|
```

The result is between 0 and 1:

- `1` means the two models make errors on the same rows.
- `0` means their error sets do not overlap.

If neither model makes an error, the union is empty and the ratio is undefined.
The default `--ec-empty-unions warn` issues one warning and stores the value as
`NaN`. The finite summary values ignore those comparisons. The other choices
are `0`, `1`, `nan`, `drop`, and `error`.

## Regression EC

For a holdout sample `s`, `df-analyze` defines a residual as:

```text
r_i(s) = prediction_i(s) - true_value(s)
```

For two models, let `a = |r_i(s)|`, `b = |r_j(s)|`, and
`z = sign(r_i(s) * r_j(s))`.

| CLI name | Value | Best value | What it compares |
|---|---:|---:|---|
| `ratio` | `min(a,b) / max(a,b)` | 1 | Similarity of residual sizes |
| `ratio_diff` | `|a-b| / (a+b)` | 0 | Relative difference between residual sizes |
| `ratio_sign` | `z * min(a,b) / max(a,b)` | 1 | Residual size and whether both predictions fall on the same side of the true value |
| `ratio_diff_sign_magnitude` | `|a-b| / (a+b)` | 0 | Residual-size difference; signed direction is saved separately |
| `intersection_union_sample` | `I(s) / U(s)` | 1 | Residual overlap for each holdout sample |
| `intersection_union_all` | `sum I(s) / sum U(s)` | 1 | Residual overlap after pooling the holdout samples |
| `intersection_union_distance` | `|r_i(s)-r_j(s)|` | 0 | Absolute disagreement between the two predictions |

For the intersection-union methods:

- If the residuals are on the same side of zero, `I = min(a,b)` and
  `U = max(a,b)`.
- If they are on opposite sides, `I = 0` and `U = a+b`.

When both residuals are zero, ratio and intersection-over-union values are 1;
ratio-difference and distance values are 0.

### Explicit ratio-diff-sign variants

The reference implementation signs the ratio difference before averaging.
Positive and negative values can therefore cancel. `df-analyze` provides two
explicit choices:

- `ratio_diff_sign_magnitude` is the default. Its main score uses the unsigned
  difference, while `ec_signed_mean` and `pair_signed_mean` keep the direction.
- `ratio_diff_sign_reference` uses the signed reference calculation directly.
  It is reported but is not included in automatic rankings.

Several regression methods are mathematically related. For example, when both
residual sizes are nonzero,
`ratio_diff = (1 - ratio) / (1 + ratio)`. Treat the methods as different views
of the same fits, not as independent votes.

`intersection_union_distance` is in the target's units and has no upper bound.
Do not compare its raw value across targets that use different units.

The default `--ec-epsilon 0` uses the formulas above. A positive epsilon can
stabilize ratio denominators near zero. The value used for a run is saved in the
metadata.

## Main options

| Option | Default | Meaning |
|---|---:|---|
| `--test-val-size` | `0.4` | Fraction or count kept as the shared holdout |
| `--error-consistency`, `--ec` | off | Run EC after feature selection and tuning |
| `--ec-folds` | `5` | Folds in each repetition; minimum 2 |
| `--ec-repetitions` | `5` | Independently shuffled K-fold repetitions; minimum 1 |
| `--ec-model-seed-mode` | `vary` | `vary` changes the model seed by fold; `fixed` reuses the base seed |
| `--ec-methods` | all seven | Regression methods to calculate |
| `--ec-holdout-role` | `test` | Controls whether ranking and EC/performance comparisons are written |
| `--ec-output-detail` | `full` | Keep `summary`, `pairwise`, or `full` output |
| `--ec-save-predictions` | off | Save holdout predictions and residual/error matrices |
| `--ec-empty-unions` | `warn` | Handle classification pairs where both error sets are empty |
| `--ec-epsilon` | `0` | Stabilize regression ratio denominators |

`--ec-model-seed-mode vary` measures changes from both the training rows and
model randomness. `fixed` keeps the model seed the same, which focuses the
comparison on changes in the training rows as far as the estimator and hardware
allow.

EC uses the external holdout already created by `df-analyze`. It does not make an
extra split. Set `--test-val-size 0.2` for an 80/20 train/holdout split.

`--ec-holdout-role test` is the safe default. It calculates EC but leaves
`model_ec_ranking.csv` and `correlation_summary.csv` with headers only. Use
`validation` only when the shared holdout is separate validation or audit data.
The option records how the holdout is being used; it does not create a new data
split.

`--ec-output-detail summary` keeps the main tables and audit files. `pairwise`
also keeps model-pair tables and plots. `full` adds sample-level tables and
diagnostics. `--ec-save-predictions` adds prediction and residual/error
matrices at any detail level.

## Runtime and hardware

Runtime grows with the number of folds, repetitions, targets, models, and
selected feature sets.

| Settings | Refits per configuration | Model pairs |
|---|---:|---:|
| 2 folds x 2 repetitions | 4 | 6 |
| Default: 5 folds x 5 repetitions | 25 | 300 |
| Classification profile: 5 x 10 | 50 | 1,225 |
| Regression profile: 5 x 50 | 250 | 31,125 |

Model fitting is usually the largest cost. Start with 2 x 2 and one model when
checking a new dataset. Increasing `--ec-repetitions` improves the stability of
the descriptive estimate, but it increases refit time in direct proportion.

`--ec-output-detail summary` reduces memory use and output size; it does not
reduce refit time. The pairwise EC calculation can use CUDA when its workload
reaches the internal threshold. Larger comparison matrices are more likely to
benefit; smaller calculations remain on NumPy because transfer overhead may
leave little improvement. Model refits follow each model's own device support,
so CPU-only models still run on the CPU.

## Reading the results

`df-analyze` creates a dataset/run folder below the selected output directory.
EC files are stored in `results/error_consistency` inside that run folder.
Start with:

- `summary.csv`: one EC summary for each target, model, feature set, and method.
- `performance_summary.csv`: predictive performance of the repeated fits on the
  shared holdout.
- `trial_failures.csv`: any refit that failed and the reason.
- `selection_guard.csv`: the declared holdout role and whether ranking and
  correlation output was enabled.

In `summary.csv`, use `optimal_value` and `optimization_direction` when reading
`ec_mean`. A larger value is not better for every regression method.

The main audit files are:

- `trial_scores.csv`: predictive score for each fitted model.
- `trial_design.csv`: repetition, fold, split seed, model seed, split sizes, and
  group-overlap checks.
- `fold_assignments.csv`: the validation fold assigned to each training row.
- `reproducibility_manifest.json`: input hashes, versions, resolved EC settings,
  and the command with sensitive values removed.

Each `<target>/<model>/<selection>_<embed-selector>/` directory contains the
tables for one configuration. Pairwise and sample-level files
depend on `--ec-output-detail`. Classification configurations always include
`leave_one_model_out.csv`. With `--ec-save-predictions`, they also include
`trial_predictions.csv` and `residual_or_error_matrix.csv`.

For multiple external test sets, the EC directories are nested under `testXX`.

## How to use EC

Use EC beside ordinary predictive performance, not instead of it. A stable
model can still be inaccurate, and an accurate model can still be unstable.

There is no EC cutoff that works for every dataset. The reported standard
deviations describe variation across dependent model pairs or holdout samples;
they are not standard errors or confidence intervals.

If the shared holdout is the final test set, report the results but do not use
them to choose a model or break a tie. If model comparison is the goal, use a
separate validation or audit holdout and set `--ec-holdout-role validation`.

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
   Application to Deep Learning." Unpublished manuscript supplied with the
   project reference material.
4. Rahman, M. M. Reference implementations:
   [simulated data](https://github.com/mostafiz67/Regression_EC_Simulation),
   [real-world data](https://github.com/mostafiz67/Regression_EC_Real_World_Data),
   [classification versus regression](https://github.com/mostafiz67/Class_EC_VS_Reg_EC),
   [MNIST](https://github.com/mostafiz67/Regression_EC_MNIST),
   [rUNet](https://github.com/mostafiz67/Regression_EC_rUnet), and
   [MIMIC-III](https://github.com/mostafiz67/Regression_EC_MIMIC-III).
5. Deep Learning Lab.
   [Classification Error Consistency](https://github.com/stfxecutables/error-consistency),
   reference implementation.
