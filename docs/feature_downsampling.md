# Large-scale feature downsampling

Feature downsampling reduces a wide matrix before df-analyze runs its usual
univariate analyses, feature selection, and model tuning. It is disabled by
default.

```powershell
python df-analyze.py --df data.csv --target outcome --mode classify `
  --feat-downsample auto --n-feat-downsample 1000
```

`--n-feat-downsample` accepts either a count or a fraction. For example, `500`
keeps at most 500 features and `0.1` keeps 10 percent.

## Methods

| Method | Uses target | Output | Suitable for indexed/sparse input |
| --- | --- | --- | --- |
| `random` | No | source columns | Yes |
| `variance` | No | source columns | Yes |
| `normalized-variance` | No | source columns | Yes |
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
range-normalized variance when a usable target or statistically valid
screening/tuning split is not available, and an F-test for ordinary supervised
data. Explicit `variance` uses raw sample variance and is therefore sensitive to
the source measurement units; prefer `normalized-variance` when raw columns have
different scales. When the feature count reaches the large-feature threshold or
the feature-to-sample ratio is at least 1,000, supervised `auto` uses
`stable-rank`: repeated subsamples of the leakage-isolated screening data are
ranked by F-test score, then combined by selection frequency and mean rank. This
avoids assigning an unvalidated equal weight to an untargeted variance score.
`rank-ensemble` remains available as an explicit exploratory method. In the
normal prepared-table path, its materialized inputs with at most 50,000 features
and a fitting matrix within the dense working-memory budget also attempt
mutual-information, linear, and LGBM members; result metadata lists only members
that actually produced valid scores. Multi-target scores are combined through
per-target ranks so targets with different score scales contribute comparably.
Equal scores receive equal average ranks; exact top-k boundary ties are resolved
reproducibly from the feature identity and configured seed.

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
used to score features. For multi-target data, both subsets must retain usable
support for every classification target and non-constant values for every
regression target. Change the screening allocation with
`--downsample-screening-fraction`. If the data cannot form disjoint subsets that
both have usable statistical support, `auto` falls back to normalized-variance
downsampling and records the reason. Each classification level must occur at
least twice in both subsets, and each regression subset must have at least three
rows. An explicitly requested supervised method stops with an explanatory error
instead. Grouped data require at least two distinct training groups and keep
every group wholly within screening or tuning; an impossible group-disjoint
split follows the same fallback/error rules and never silently reuses groups
across the two phases.

## Extremely wide numeric tables

Use `--large-feature-mode` when the source table is numeric and the ordinary
preparation pipeline would be too expensive:

```powershell
python df-analyze.py --df wide.parquet --target outcome --mode classify `
  --large-feature-mode --feat-downsample rank-ensemble `
  --n-feat-downsample 1000
```

This mode splits rows and scores columns before materializing the selected
training and test matrices. Categorical and ordinal predictors are not accepted
because their encoding can change the feature dimension. Only indexed-safe
methods from the table above are available. The source table is still loaded as
a pandas DataFrame, so it must fit in memory. Only rows in the current train/test
split and the selected columns are copied into normal preprocessing, with a
peak-working-memory guard. Use SVMlight input when the source matrix itself is
too large to materialize densely.

The ensemble methods in large-feature mode use only the scale-invariant,
column-chunked range-normalized-variance and F-test members. They do not use
raw, pre-preprocessing linear coefficients.

## SVMlight input

Files ending in `.svm`, `.svmlight`, `.libsvm`, or `.binary` are recognized,
including gzip, bzip2, and xz compression. The matrix stays sparse during
scoring and only selected columns are converted to the final dense matrices.
The selected numeric columns are then scaled with a training-fitted,
zero-preserving maximum-absolute-value transform. This avoids erasing rare
non-zero events in sparse binary or count features. Dense materialization uses a
peak-working-memory guard; reduce `--n-feat-downsample` when the selected
train/test matrices would exceed it.

```powershell
python df-analyze.py --df wide.svmlight --target outcome --mode classify `
  --feat-downsample f-test --n-feat-downsample 500
```

Auto mode recognizes the standard sparse suffixes and performs a bounded
content check for files with nonstandard names. This includes extensionless text
and gzip, bzip2, or xz inputs such as `log1p.E2006.train.bz2`. Use
`--input-format svmlight` to make the format explicit. `--svmlight-index-base`
can be `auto`, `zero`, or `one`; generated feature names retain the resolved
source index base. Because a file without feature index 0 could be either
zero-based or one-based, use an explicit index base whenever the producer's
convention is known. SVMlight input requires feature downsampling and exactly
one target. It cannot be combined with `--large-feature-mode`.

For example, the official E2006 train/test files have more than 4.2 million
one-based features and no standard SVMlight filename suffix:

```powershell
python df-analyze.py `
  --df-train C:\data\log1p.E2006.train `
  --df-tests C:\data\log1p.E2006.test `
  --df-tests-method list `
  --target target --mode regress `
  --input-format svmlight --svmlight-index-base one `
  --feat-downsample auto --n-feat-downsample 500 `
  --regressors sgd dummy --htune-trials 10 `
  --outdir .\e2006_results
```

To combine CRUSH-scale imaging values with the ordinary clinical spreadsheet,
pass one row-aligned sidecar per SVMlight input:

```powershell
python df-analyze.py --df crush.svmlight --target diagnosis --mode classify `
  --svmlight-metadata participants.tsv `
  --svmlight-feature-map crush_features.csv `
  --svmlight-sample-id-column participant_id `
  --categoricals sex site --drops participant_id `
  --feat-downsample stable-rank --n-feat-downsample 1000
```

Sidecars may be CSV, TSV, JSON, or Parquet. Their row count and order must match
the corresponding sparse file exactly. If the target column is present,
df-analyze verifies one-to-one class correspondence (or numeric equality for
regression) against the SVMlight target; otherwise it adds the SVMlight target.
All sidecars must have the same schema. Clinical columns are appended only after
imaging downsampling and then use the normal df-analyze inference, encoding, and
cleaning pipeline. A sidecar also enables `--grouper`, `--categoricals`,
`--ordinals`, and `--drops`; groups remain disjoint during both initial
holdout splitting and supervised screening.

For strong row-identity validation, add comments to the corresponding SVMlight
rows, for example `1 4:0.2 91:1 # participant_id=sub-0001`, and pass
`--svmlight-sample-id-column participant_id`. The comments are checked exactly
against that clinical column before splitting; the ID column is automatically
dropped unless it is the configured grouper. Without this option, target
correspondence detects many ordering errors but cannot detect permutations
within the same target class.

The optional feature map requires `feature_index` and `feature_name` columns.
Indices use the configured SVMlight index base. An optional `protected` column
accepts `true`, `yes`, `y`, or `1`; those imaging features are always retained
by the downsampling step and count toward `--n-feat-downsample`. Additional
mapped or generated names can be protected with
`--downsample-protected-features`. Protected features can still be removed later
if normal preprocessing proves that they are constant or otherwise unusable.

Single-file train/test splitting retains one source CSR instead of copying both
partitions. External LODO materializes selected dense test blocks sequentially
and does not stack million-column sparse matrices. The main analysis first
scans dimensions and labels without loading predictor matrices, then retains at
most the current training CSR and one current test CSR. Generated feature names
remain lazy.

The downsampling JSON and Markdown report time input layout/target scanning,
training CSR loading, feature scoring, external test CSR loading, and selected
feature materialization/scaling separately. The reported score-chunk memory
budget is not a whole-process RSS limit: it excludes source CSR matrices,
temporary sparse conversions, full-length score/rank vectors, and selected dense
outputs.

## Outputs

Each split writes the following files under `features/downsampling`:

- `downsampling_report.md` and `downsampling.json`
- `selected_features.csv`
- `feature_scores.csv` when the method produces scores

For external validation, fold suffixes are added (for example,
`downsampling_report_00.md`). The preparation report lists training and external
test input/final shapes separately so test rows are not mistaken for dropped
training samples.

For SVMlight input, the CSV files keep `feature_index` as the internal
zero-based matrix position and also include `source_feature_index` in the
configured source index base. The JSON records `source_index_base`,
`selected_source_indices`, and (for bounded score summaries)
`score_source_feature_indices`. Use the source-index fields when joining results
back to a CRUSH or other SVMlight feature map.

By default only the leading 500 scores are saved for very wide input. Pass
`--downsample-save-scores` to stream all of them in bounded chunks, including
unavailable NaN and negative-infinity scores. `feature_scores.csv` includes
`score_valid` and `invalid_reason` so its row count always equals the source
feature count when full saving is requested. Use
`--skip-full-prepared-save-before-downsample` to avoid writing the full prepared
matrix in the ordinary table path.
