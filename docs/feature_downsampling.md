# Large-scale feature downsampling

Feature downsampling reduces a wide matrix before df-analyze runs its usual
univariate analyses, feature selection, and model tuning. It is disabled by
default.

```shell
python df-analyze.py \
    --df data.csv \
    --target outcome \
    --mode classify \
    --feat-downsample auto \
    --n-feat-downsample 1000
```

`--n-feat-downsample` accepts either a count or a fraction. For example, `500`
keeps at most 500 features and `0.1` keeps 10 percent.

For a first run, use `auto`. Use another method only when you have a reason to
control how features are ranked or projected.

## Choosing an input path

- Use the normal table path for data that fits comfortably in memory.
- Add `--large-feature-mode` for an all-numeric table that fits in memory but
  is too wide for normal preprocessing.
- Use SVMlight for a sparse matrix that should not be loaded as a dense table.

All three paths use only training data to choose features. The holdout and any
external test sets are not used for feature ranking.

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

`auto` works as follows:

1. If the data already has no more than the requested number of features, keep
   it unchanged.
2. For ordinary supervised data, rank features with an F-test.
3. For extremely wide data (or at least 1,000 features per sample), use
   `stable-rank`. This repeats the F-test on several screening subsamples and
   combines the selection frequency and average rank.
4. If a supervised screening split cannot be made, use
   `normalized-variance`.

Explicit `variance` uses raw sample variance, so a change in measurement units
can change the ranking. Use `normalized-variance` when columns have different
scales.

`rank-ensemble` is an exploratory alternative. On a prepared table with at
most 50,000 features, and when the fitting matrix stays within the memory
limit, it may combine F-test, normalized-variance, mutual-information, linear,
and LGBM scores. The report lists the members that actually ran.

For multiple targets, scores are converted to per-target ranks before they are
combined. This prevents a target with numerically larger scores from dominating
the result. Equal scores receive equal ranks; ties at the final top-k boundary
are broken reproducibly from the feature name and seed.

`stable-rank` is a compatibility name for repeated-subsample F-test rank
aggregation; it does not refer to the matrix stable-rank quantity.

Variance, F-test, and ensemble scores are computed in column chunks. The
requested `--downsample-chunk-size` is reduced automatically when a chunk would
exceed the internal working-memory limit.

Downsampling reduces computation, but it can discard useful features. Variance
and F-test screening can miss a feature that is useful only through an
interaction with another feature. When the full analysis is practical, compare
its holdout result with the downsampled run.

## Leakage control

Supervised methods divide each training fold into two parts. One part ranks the
features; the other tunes the model. The tuned model is then refit on the full
training fold. Change the first part with `--downsample-screening-fraction`.

Both parts must contain enough data for the selected task. Every classification
level must appear at least twice in both parts. Each regression part must have
at least three rows; in a multi-target run, every regression target must also
be finite and non-constant in both parts.

For grouped data, a group stays wholly in one part and at least two training
groups are required. If these splits cannot be made, `auto` records the reason
and switches to `normalized-variance`. A supervised method selected by name
stops with an error instead.

## Extremely wide numeric tables

Use `--large-feature-mode` when the source table is numeric and the ordinary
preparation pipeline would be too expensive:

```powershell
python df-analyze.py --df wide.parquet --target outcome --mode classify `
  --large-feature-mode --feat-downsample rank-ensemble `
  --n-feat-downsample 1000
```

This mode splits rows and scores columns before it builds the selected training
and test matrices. Predictors must be numeric and finite; categorical and
ordinal predictors need the normal preparation path. Only methods marked
suitable for indexed input in the table above can be used.

The complete source table is still loaded as a pandas DataFrame and must fit in
memory. Use SVMlight when the source matrix needs to remain sparse.

In this mode, ensemble methods use only chunked `normalized-variance` and
F-test scores.

## SVMlight input

Files ending in `.svm`, `.svmlight`, `.libsvm`, or `.binary` are recognized,
including gzip, bzip2, and xz compression. The matrix stays sparse during
scoring and only selected columns are converted to the final dense matrices.
The selected numeric columns are scaled from the training data without turning
sparse zeroes into non-zero values. If the selected dense train/test matrices
would exceed the memory limit, reduce `--n-feat-downsample`.

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

### Adding Clinical Data and Feature Names

To combine CRUSH-scale imaging values with an ordinary clinical table, pass one
row-aligned sidecar per SVMlight input:

```powershell
python df-analyze.py --df crush.svmlight --target diagnosis --mode classify `
  --svmlight-metadata participants.tsv `
  --svmlight-feature-map crush_features.csv `
  --svmlight-sample-id-column participant_id `
  --categoricals sex site --drops participant_id `
  --feat-downsample stable-rank --n-feat-downsample 1000
```

Sidecars may be CSV, TSV, JSON, or Parquet. Their row count and order must match
the corresponding sparse file exactly, and all sidecars must have the same
columns. If a sidecar contains the target, df-analyze checks it against the
SVMlight target. Otherwise, the SVMlight target is added to the clinical data.

Clinical columns are added after imaging downsampling, then pass through normal
type inference, encoding, and cleaning. A sidecar also enables `--grouper`,
`--categoricals`, `--ordinals`, and `--drops`.

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

### Memory and Timing Reports

For a single input file, the train/test split shares one sparse source matrix.
For external leave-one-dataset-out (LODO) validation, test matrices are loaded
one at a time. The main analysis first reads dimensions and labels, then keeps
only the current training matrix and one current test matrix in memory.

The JSON and Markdown reports time input scanning, sparse-matrix loading,
feature scoring, external test loading, and selected-feature conversion
separately. The score-chunk memory budget covers only the current scoring chunk,
not the whole process.

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
