# Program Outputs

## Output Directories

- **inspection**: inferred cardinality (categorical, ordinal, continuous) and
  types for features
- **prepared**: summary of data preparation (compute times, final data shape
  after encoding and drops)
- **features**: information about the features
- **features/downsampling**: selected source columns or projected component
  metadata, score tables, runtimes, and the resolved downsampling method
- **features/associations**: univariate (statistical) associations of each
  feature with the target variable
- **features/predictions**: predictive performance on the target for each
  feature in isolation
- **selection**: information about features selected and methods used during
  feature selection
- **selection/embed**: embedded feature selection (via LASSO or LightGBM)
  details
- **selection/filter**: feature selection based on univariate associations or
  predictive performance
- **selection/wrapper**: feature selection based on wrapper methods
  (currently only step-up and backward)
- **results**: final tables of predictive performance after feature selection
  and tuning
- **results/error_consistency**: repeated K-fold error consistency summaries,
  per-configuration diagnostics, trial scores, and plots
- **tuning**: (currently unused) params for each tuned model and each
  selection method

For multi-target runs, target-specific feature reports are stored in subfolders
named after each target. The `prepared` directory also contains the target and
split audits. Final results include:

- `final_performances_per_target.csv`
- `results_report_target_<target>.md`
- `performance_long_table_per_target.csv`
- `multitarget_split_report.md` (or one file per external-test fold)

Multi-target regression tuning normalizes error scores per target before
aggregation. Final MAE, MSE, and RMSE values are not normalized; when targets
have different units, compare their per-target tables or the scale-free R²
metrics.

Final holdout cross-validation uses at most five folds. Grouped runs reduce the
fold count when the holdout contains fewer than five distinct groups, while
preserving group-disjoint folds. `final_cv_folds` records the count actually
used. The value column named `5-fold` is retained for output compatibility and
must be interpreted using `final_cv_folds`. Fewer than five folds also produces
a warning because the resulting estimate can be unstable.

When `--error-consistency` is enabled, `results/error_consistency` contains:

See [Error consistency and repeated K-fold design](error_consistency.md) for the
formulas, aggregation rules, randomness controls, and interpretation limits.

- `summary.csv`: EC values for every target, model, selection, and EC method,
  together with `scientific_status`, `reference_url`, and `inferential_status`.
  Classification error IoU identifies its published reference; regression
  residual-consistency methods are marked as experimental descriptive diagnostics.
- `performance_summary.csv`: mean and sample standard deviation of the repeated
  K-fold models on the common holdout
- `trial_scores.csv`: predictive score from every repeated fold model
- `trial_design.csv`: repetition/fold split and model seeds, split sizes, and
  group-overlap audit for successful configurations; grouped configurations that
  require a non-grouped fallback are skipped
- `fold_assignments.csv`: exact validation-fold assignment of every training-row
  position in every repetition (the training fold is its complement)
- `correlation_summary.csv` and `model_ec_ranking.csv`: target-specific
  performance/stability diagnostics
- `target_ec_trend.csv`: target-level EC trend summaries
- `metadata.json` and `README.md`: run metadata and interpretation guidance
- `plots/`: EC distributions, EC/performance comparisons, and a correlation
  heatmap when the required data are available
- `<target>/<model>/<selection>_<embed-selector>/`: pairwise matrices, sample diagnostics,
  trial scores, exact fold assignments, difficult/unstable sample tables,
  optional group diagnostics, and pairwise plots. Pairwise values label
  within- versus between-repetition comparisons. `--ec-save-predictions`
  additionally writes `trial_predictions.csv` and
  `residual_or_error_matrix.csv`.

When adaptive error and EC are both enabled,
`results/adaptive_error/tables` also contains
`risk_stability_report.csv`, `risk_stability_summary.csv`, and
`risk_stability_skipped.csv`.

For multiple external test sets these files are nested under `testXX`. Error
consistency holds feature selection and tuned hyperparameters fixed; it measures
refit stability and is not a nested re-selection analysis.

`ec_model_pair_sd` is the sample standard deviation of model-pair EC means.
`ec_pooled_value_sd` (and the legacy `ec_sd`) pools pair-by-sample values for
samplewise regression methods. These are descriptive dispersions of dependent
comparisons, not standard errors or confidence intervals.

## Output Files

## Makrkdown Reports

- **`associations_report.md`**
  - summarizes the (statistical) associations between each feature and the
    target variable
  - columns (classification):
    - **mut_info**: `sklearn.feature_selection` mutual information, either
      classification or regression variant as appropriate
    - **t**: independent samples t-test
    - **t_p**: p-value for t-test
    - **U**: Mann-Whitney U statistic
    - **U_p**: p-value for Mann-Whitney U statistic
    - **W**: Brunner Munzel W
      (https://en.wikipedia.org/wiki/Brunner_Munzel_Test). Same idea as
      Mann-Whitney U, but less assumptions.
    - **W_p**: p-value for Brunner Munzel test
    - **cohen_d**: Cohen's d
    - **AUROC**: area under the ROC curve (defined as
      [here](https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=1188631305#Area-under-curve_(AUC)_statistic_for_ROC_curves))
    - **corr**: Pearson's correlation coefficient
    - **corr_p**: p-value for Pearson's correlation coefficient
  - columns (regression):
    - **mut_info**: `sklearn.feature_selection` mutual information, either
      classification or regression variant as appropriate
    - **pearson_r**: Pearson's correlation coefficient
    - **pearson_p**: p-value for Pearson's correlation coefficient
    - **spearman_r**: Spearman's correlation coefficient
    - **spearman_p**: p-value for Spearman's correlation coefficient
    - **F**: `sklearn.feature_selection.f_regression`, i.e. F-statistic
    - **F_p**: p-value for F-statistic above
    - **H**: Kruskal-Wallace H, i.e. one-way ANOVA on ranks
      (https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=1193273201)
    - **H_p**: p-value for Kruskal-Wallace H

- **`predictions_report.md`**
  - summarizes the univariate predictive performance of each feature on the
    target variable
- **`short_inspection_report.md`**
  - summarizes the inferred cardinalities (categorical, ordinal, continuous)
    of each feature
  - also summarizes if any features have been identified as unusable (e.g.
    timeseries features, identifiers, or constant features)
- **`preparation_report.md`**
  - summarizes changes to data shape after encoding of NaNs and categoricals
- **`results_report.md`**
  - summarizes predictive performance of the tuned models based on each
    feature selection procedure
- **`embedded_selection_report.md`**
  - summarizes which features were selected by the embedded method, as well
    as the selecting model and overall feature importances generated by this
    model
- **`association_selection_report.md`**
  - summarizes which features were selected based on their statistical
    associations, and which measure of association was used
- **`prediction_selection_report.md`**
  - summarizes which features were selected based on their univariate
    predictive performance, and which performance metric was used
- **`wrapper_selection_report.md`**
  - summarizes which features were selected based on the wrapper selection
    method, and the wrapper model and method used


## Tables

Raw data (unrounded) used to generate the tables in each report are present
in `.csv` files in the same folder as the Markdown (`.md`) reports.

Compressed forms of each table are also often saved in `.parquet` [file
format](https://parquet.apache.org/docs/file-format/) in each directory.
