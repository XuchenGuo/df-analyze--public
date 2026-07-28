# Large-scale feature downsampling

Feature downsampling reduces a wide matrix before df-analyze runs its usual
univariate analyses, feature selection, and model tuning. It is disabled by
default.

```powershell
python df-analyze.py --df data.csv --target outcome --classify `
  --feat-downsample auto --n-feat-downsample 1000
```

`--n-feat-downsample` accepts either a count or a fraction. For example, `500`
keeps at most 500 features and `0.1` keeps 10 percent.

## Methods

| Method | Uses target | Output | Suitable for indexed/sparse input |
| --- | --- | --- | --- |
| `random` | No | source columns | Yes |
| `variance` | No | source columns | Yes |
| `f-test` | Yes | source columns | Yes |
| `mutual-info` | Yes | source columns | No |
| `linear` | Yes | source columns | No |
| `lgbm` | Yes | source columns | No |
| `svd` | No | components | No |
| `sparse-rp` | No | components | No |
| `rank-ensemble` | Yes | source columns | Yes |
| `selector-ensemble` | Yes | source columns | Yes |
| `stable-rank` | Yes | source columns | Yes |

`auto` uses no downsampling when the requested dimension is already available,
variance when a usable target is not available, and an F-test for ordinary
supervised data. When the feature count reaches the large-feature threshold or
the feature-to-sample ratio is at least 1,000, supervised `auto` uses
`rank-ensemble`, combining range-normalized variance and F-test ranks instead of
relying on raw F scores alone. Both members are invariant to non-zero linear
rescaling, which avoids favoring source columns solely because they use larger
numeric units. Materialized inputs with at most 50,000 features also attempt
mutual-information, linear, and LGBM members; result metadata lists only members
that actually produced valid scores. Multi-target scores are combined through
per-target ranks so targets with different score scales contribute comparably.

`stable-rank` is a compatibility name for repeated-subsample F-test rank
aggregation; it does not refer to the matrix stable-rank quantity.

Variance, F-test, and ensemble scores are computed in column chunks. The
requested `--downsample-chunk-size` is reduced automatically when a chunk would
exceed the internal working-memory limit.

These methods reduce computation; they do not guarantee a globally optimal
predictive feature set. In particular, variance and univariate F-test screening
can miss features whose signal exists only through interactions. Compare final
holdout performance with a no-downsampling baseline whenever the full baseline is
computationally feasible, and use repeated or ensemble screening when selection
stability matters.

## Leakage control

Supervised downsampling fits on a screening subset of each training fold. Model
hyperparameters are tuned on the disjoint remainder, then the tuned estimator
is refit on the full training fold. Holdout and external test rows are never
used to score features. Change the screening allocation with
`--downsample-screening-fraction`. If the data cannot form disjoint subsets that
both contain every classification target level, `auto` falls back to variance
downsampling and records the reason. An explicitly requested supervised method
stops with an explanatory error instead. Grouped data require at least two
distinct training groups and keep every group wholly within screening or
tuning; an impossible group-disjoint split follows the same fallback/error
rules and never silently reuses groups across the two phases.

## Extremely wide numeric tables

Use `--large-feature-mode` when the source table is numeric and the ordinary
preparation pipeline would be too expensive:

```powershell
python df-analyze.py --df wide.parquet --target outcome --classify `
  --large-feature-mode --feat-downsample rank-ensemble `
  --n-feat-downsample 1000
```

This mode splits rows and scores columns before materializing the selected
training and test matrices. Categorical and ordinal predictors are not accepted
because their encoding can change the feature dimension. Only indexed-safe
methods from the table above are available. The source table is still loaded as
a pandas DataFrame, so it must fit in memory. Use SVMlight input when the source
matrix itself is too large to materialize densely.

## SVMlight input

Files ending in `.svm`, `.svmlight`, `.libsvm`, or `.binary` are recognized,
including gzip, bzip2, and xz compression. The matrix stays sparse during
scoring and only selected columns are converted to the final dense matrices.
The selected numeric columns are then normalized using parameters fitted on the
training partition.

```powershell
python df-analyze.py --df wide.svmlight --target outcome --classify `
  --feat-downsample f-test --n-feat-downsample 500
```

Use `--input-format svmlight` for another suffix. `--svmlight-index-base` can be
`auto`, `zero`, or `one`; generated feature names retain the resolved source
index base. SVMlight input requires feature downsampling and exactly one target.
It cannot be combined with `--large-feature-mode`, a grouper, categorical or
ordinal declarations, or dropped-column declarations.

## Outputs

Each split writes the following files under `features/downsampling`:

- `downsampling_report.md` and `downsampling.json`
- `selected_features.csv`
- `feature_scores.csv` when the method produces scores

By default only the leading 500 scores are saved for very wide input. Pass
`--downsample-save-scores` to save all of them. Use
`--skip-full-prepared-save-before-downsample` to avoid writing the full prepared
matrix in the ordinary table path.
