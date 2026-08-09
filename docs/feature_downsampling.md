# Feature downsampling

Feature downsampling reduces very wide predictor matrices before the ordinary
preparation, feature-selection, and tuning stages. It is disabled by default.

The first implementation intentionally supports only two ranking methods:

| Method | Uses target | Output |
|---|---:|---|
| `normalized-variance` | No | Named source columns |
| `f-test` | Yes | Named source columns |

`normalized-variance` ranks each feature by variance divided by its squared
range. Translation and nonzero rescaling therefore do not change the score.
Constant and otherwise invalid features are not eligible for selection.

`f-test` ranks features on a screening subset of the training data. The
remaining training rows are reserved for model tuning. Holdout and external
test rows never contribute to feature scores. With grouped data, complete
groups remain together. If a valid screening/tuning split cannot be made, the
run fails; it does not switch methods.

## Basic use

```shell
python df-analyze.py \
  --df wide.parquet \
  --target outcome \
  --mode classify \
  --feat-downsample f-test \
  --n-feat-downsample 1000 \
  --outdir ./wide_results
```

`--n-feat-downsample` accepts a positive count or a fraction in `(0, 1]`.
Protected features count toward this limit. They are removed from each ranking
before the remaining top-k features are selected, so a protected feature cannot
change the ordering of non-protected features.

## Large numeric tables

Use `--large-feature-mode` when the source table fits in memory but is too wide
for ordinary preparation:

```shell
python df-analyze.py \
  --df wide.parquet \
  --target outcome \
  --mode classify \
  --large-feature-mode \
  --feat-downsample f-test \
  --n-feat-downsample 1000 \
  --outdir ./wide_results
```

Predictors must be numeric and finite. Rows are split and columns are scored
before the selected dense matrices are materialized.

## SVMlight input

SVMlight matrices stay sparse during scoring:

```shell
python df-analyze.py \
  --df wide.svmlight \
  --target outcome \
  --mode classify \
  --feat-downsample normalized-variance \
  --n-feat-downsample 500 \
  --outdir ./svmlight_results
```

Supported suffixes include `.svm`, `.svmlight`, `.libsvm`, and `.binary`, with
gzip, bzip2, or xz compression. Use `--svmlight-index-base zero|one` when the
producer's index convention is known. Clinical sidecars and feature maps can be
provided with `--svmlight-metadata` and `--svmlight-feature-map`.

## Outputs

Results are stored under `features/downsampling`:

- `downsampling_report.md`
- `downsampling.json`
- `selected_features.csv`
- `feature_scores.csv`, when scores are saved

Feature downsampling is a screening step and may discard useful interactions.
When practical, compare the result with a run that does not downsample.
