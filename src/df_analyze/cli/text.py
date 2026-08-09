from df_analyze.enumerables import (
    ClsScore,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    RegScore,
)

USAGE_STRING = """

The df-analyze program can be used in one of two modes: CLI mode, and
spreadsheet mode. In spreadsheet mode, df-analyze options are specified in a
special format at the top of a spreadsheet or .csv file, and spreadsheet
columns are given specific names to identify targets, continuous features,
and categorical features. In spreadsheet mode, only a single argument needs
to be passed, which is the path to the df-analyze formatted spreadsheet:

    python df-analyze.py --spreadsheet my_formatted_sheet.xlsx

Good defaults are chosen so that likely the only arguments you would wish to
specify manually are:

    --target (required)
    --mode (required)
    --outdir

"""

USAGE_EXAMPLES = """
USAGE EXAMPLE (run from the repository root with the environment activated):

    python df-analyze.py \\
        --df weather_data.json \\
        --target temperature \\
        --categoricals weekday,season \\
        --ordinals rainfall_mm \\
        --drops date,sample_id \\
        --mode=regress \\
        --regressors=elastic lgbm knn \\
        --feat-select wrap embed filter \\
        --embed-select linear \\
        --wrapper-select step-up \\
        --wrapper-model linear \\
        --norm robust \\
        --nan median \\
        --n-feat-filter 10 \\
        --n-feat-wrapper 10 \\
        --test-val-size=0.25 \\
        --outdir='./results'

"""


SHEET_HELP_STR = """
The path to the formatted spreadsheet to analyze.

Currently only spreadsheets saved as either `.xlsx` or `.csv` are supported.

If your data is saved as a Pandas `DataFrame`, it must have shape
`(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
column holding the target variable (or feature) can be specified by the
`--target` / `-y` argument, but is "target" by default if such a column name
exists, or the last column if it does not.

For more details, see the README at https://github.com/stfxecutables/df-analyze.

"""

DF_FILETYPES = """
Tables may be saved as `.parquet`, `.xlsx`, `.json`, or `.csv`. Files exported
with a Pandas `DataFrame.to_*` method are preferred. SVMlight text input is also
supported. Auto mode recognizes common sparse-file suffixes. For other
filenames, it reads a short prefix to detect SVMlight, including gzip, bzip2,
and xz-compressed files.

If your data is saved as a Pandas `DataFrame`, it must have shape
`(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
column holding the target variable (or feature) can be specified by the
`--target` / `-y` argument, but is "target" by default if such a column name
exists, or the last column if it does not."""

DF_HELP_STR = f"""
The dataframe to analyze. Cannot be used with `--df-train` or `--df-tests` arguments.
{DF_FILETYPES}

"""

DF_TRAIN_HELP_STR = f"""
The training dataframe to analyze. Cannot be used with `--df` argument.
{DF_FILETYPES}

"""

DF_TESTS_HELP_STR = f"""
The dataframes to use for testing. How these are used depends on the
--df-tests-method option. Cannot be used with `--df` argument.
{DF_FILETYPES}

"""

DF_TEST_SETS_METHOD_HELP_STR = """
The manner in which to use the specified test sets for validation. Available
options are:

  lodo        Leave-One-Dataset-Out. Assuming --df-train=file0, and that
              --df-tests=file1,file2,...,fileN, then df-analyze runs N+1 full
              runs of the feature selection, tuning, and validation pipeline,
              for the sets:

              X_train_0={{file0}}, X_test_0=concat({{file1, file2, ..., fileN}})
              X_train_1={{file1}}, X_test_1=concat({{file0, file2, ..., fileN}})
                                         ...
              X_train_N={{fileN}}, X_test_N=concat({{file0, file2, ..., fileN-1}})

              Note: In the case of a single test file, this is just 2-fold. For
              more than 2 test files, expect this option to potentially be very
              expensive on anything other than small datasets.

              Note: In the case of N test files, this produces N+1 feature
              selection reports, and N+1 final performance summary tables.

  list        Perform feature selection and tuning only once on the file passed
              in to --df-train, and then run validation (holdout, internal
              k-fold) on each of the files passed into --df-tests

              Note: In the case of N test files, this produces ONE feature
              selection report, and N final performance summary tables.

"""

SEED_HELP_STR = """
Seed for random splits if using `--spreadsheet` or `--df` arguments. Options:

  random      Do not use a fixed seed (np.random.default_rng() will be called
              to create RNGs where needed).

  default     Use the fixed, default constant seed. With this option, (or if
              the `--seed` option is unspecified), the same splits are created
              across df-analyze invocations.

  [integer]   Integer seed for np.random.default_rng in splitting operations.

"""

SEP_HELP_STR = """
Separator used in .csv files. Default ",".

"""

TARGET_HELP_STR = """
The (string) name of the target variable for either regression or
classification.

"""

TARGETS_HELP_STR = """
Comma-separated target columns for a multi-target run, for example
`--targets outcome_a,outcome_b`. All targets in one run must use the same
`--mode`: use `classify` for categorical targets or `regress` for numeric
targets. Rows missing any target are removed. This option takes precedence over
`--target`.

"""

GROUP_HELP_STR = """
The (string) name of the grouping variable (if one is present) which will be
used to ensure samples within the same group do not end up in both train and
test splits. I.e. the name of the feature that will be passed into scikit-learn
GroupStratifiedKFold.

Grouped analyses stop if adequate group-disjoint classification folds cannot be
created; they do not silently fall back to row-wise splitting.

"""

CATEGORICAL_HELP_STR = """
A string or list of strings, e.g.

    --categoricals sex,gender,ancestry,education

that specifies which features will be treated as categorical regardless of the
number of levels or format of the data. If during data cleaning categorical
variables are detected that are NOT specified by the user, a warning will be
raised.

Commas currently must be used to separate feature names: spaces will be
considered to be part of the feature name, i.e. if you pass:

    --categoricals sex,gender, ancestry,education

this will parse as the features "sex", "gender", " ancestry" (note space!),
and "education".

"""

ORDINAL_HELP_STR = """
A string or list of strings, e.g.

    --ordinals star_rating,number_of_purchases,times_signed_in

that specifies which features will be treated as ordinal regardless of the
number of levels or format of the data. If during data cleaning ordinal
variables are detected that are NOT specified by the user, a warning will be
raised. If the values of the specified variables cannot be interpreted as
integers, then df-analyze will exit with an error.

See documentation for `--categoricals` re: the use of commas to list features.

"""

DROP_HELP_STR = """
A string or list of strings, e.g.

    --drops subject_id,location_id,useless_feature1,useless_feat2

that specifies which features will be removed from the data and not considered
for any inspection, description or univariate analysis, and which will not be
included in any feature selection, model tuning, or final predictive models.

See documentation for `--categoricals` re: the use of commas to list features.

"""

MODE_HELP_STR = """
If "classify", do classification. If "regress", do regression.

"""

DEVICE_HELP = """
Choose how supported work uses an accelerator. `auto` (recommended) checks each
selected component before execution: it uses CUDA when that component's backend
is available, MPS for GANDALF on a supported Mac, and otherwise CPU. `cpu`
disables accelerator checks. `cuda` requires CUDA for selected work that supports
it; missing backends and runtime CUDA errors stop the run, while CPU-only work
remains on the CPU. The startup plan shows the resolved device for every selected
component.
"""

CLASSIFIER_CHOICES = DfAnalyzeClassifier.choices()

CLASSIFIER_DEFAULTS = DfAnalyzeClassifier.defaults()

REGRESSOR_CHOICES = DfAnalyzeRegressor.choices()

REGRESSOR_DEFAULTS = DfAnalyzeRegressor.defaults()


CLS_HELP_STR = f"""
The list of classifiers to use when comparing classification performance.
Can be a list of elements from: [{" ".join(sorted(CLASSIFIER_CHOICES))}].
Defaults are: [{" ".join(CLASSIFIER_DEFAULTS)}].

  catboost    CatBoost classifier.

  xgb         XGBoost classifier.

  tabpfn      Versioned TabPFN foundation-model classifier.

  dtree       scikit-learn DecisionTreeClassifier.

  et          scikit-learn ExtraTreesClassifier.

  knn         KNN classifier (scikit-learn on CPU or PyTorch on CUDA).

  lgbm        LightGBM boosted decision tree classifier.

  rf          LightGBM random forest classifier.

  lr          scikit-learn LogisticRegression.

  sgd         scikit-learn SGDClassifier.

  mlp         Modern multi-layer perceptron implemented in skorch/PyTorch.

  kan         Kolmogorov-Arnold Network implemented with PyKAN/skorch.

  svm         Registered for compatibility; currently disabled.

  gandalf     Gated Adaptive Network for Deep Automated Learning of
              Features for Tabular Data: https://arxiv.org/abs/2207.08548

  dummy       scikit-learn DummyClassifier.

"""

REG_HELP_STR = f"""
The list of regressors to use when comparing regression model performance.
Can be a list of elements from: [{" ".join(sorted(REGRESSOR_CHOICES))}].
Defaults are: [{" ".join(REGRESSOR_DEFAULTS)}].

  catboost    CatBoost regressor.

  xgb         XGBoost regressor.

  tabpfn      Versioned TabPFN foundation-model regressor.

  dtree       scikit-learn DecisionTreeRegressor.

  et          scikit-learn ExtraTreesRegressor.

  knn         KNN regressor (scikit-learn on CPU or PyTorch on CUDA).

  lgbm        LightGBM boosted decision tree regressor.

  rf          LightGBM random forest regressor.

  elastic     scikit-learn ElasticNet.

  sgd         scikit-learn SGDRegressor.

  mlp         Modern multi-layer perceptron implemented in skorch/PyTorch.

  kan         Kolmogorov-Arnold Network implemented with PyKAN/skorch.

  svm         Registered for compatibility; currently disabled.

  gandalf     Gated Adaptive Network for Deep Automated Learning of
              Features for Tabular Data: https://arxiv.org/abs/2207.08548

  dummy       scikit-learn DummyRegressor.
"""

FEAT_SELECT_HELP = """
The feature selection method(s) to use. Available options are:

  filter      Select features based on their univariate relationships to the
              target variables.

  embed       Select features using a model with implicit feature selection,
              e.g. an L1-regularized model or decision tree. For available
              models, see `--embed-select`.

  wrap        Select features by recursive model evaluation, currently either
              step-up (forward) feature selection, or step-down (backward)
              feature elimination.

  none        Do not perform any selection.

NOTE: Multiple selection options can be compared by passing each option, e.g.

  python df-analyze.py [...] --feat-select filter embed wrap

NOTE: Feature selection currently uses a training split of the full data
provided in the `--df` or `--spreadsheet` argument to `df-analyze.py`. This
is to prevent double-dipping / circular analysis that can result in
(extremely) biased performance estimates.

"""

FEAT_DOWNSAMPLE_HELP = """
Reduce a wide feature matrix before the usual df-analyze feature selection.
`normalized-variance` is target-independent and scale invariant. `f-test`
uses a disjoint screening subset. `none` disables this step.
"""

N_FEAT_DOWNSAMPLE_HELP = """
Maximum number or fraction of features retained by feature downsampling.
"""

DOWNSAMPLE_SCREENING_HELP = """
Fraction of each training fold used to rank features. The remaining training
rows are used for model tuning, so the same rows are not used for both steps.
"""

LARGE_FEATURE_MODE_HELP = """
Downsample numeric columns before normal preprocessing. Use this for a table
that fits in memory but is too wide for the normal preparation path.
"""

EMBED_SELECT_MODEL_HELP = """
Model(s) to use for embedded feature selection. In either case, model is
hyperparameter tuned on the training split so that only the best-fitting model
is used for embedded feature selection. Supported models are:

  linear      SGDRegressor or SGDClassifier, in both cases with L1
              regularization.

  lgbm:       LightGBM regressor or classifier, depending on task.

  none:       Do not select features based on any model. Always included by
              by default.

Note: Embedded feature selection is currently performed via scikit-learn's
`feature_selection.SelectFromModel` with the default threshold of "mean",
meaning the number of features selected is automatic and not currently
configurable.

"""

WRAP_SELECT_HELP = """
Wrapper-based feature selection method, i.e. method/optimizer to use to
search feature-set space during wrapper-based feature selection. Currently
only (recursive) step-down and step-up methods are supported, but future
versions may support random subset search, LIPO
(http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html,
https://arxiv.org/abs/1703.02628) and evolutionary algorithms such as
particle-swarm optimization and genetic algorithms (since feature selection
is just a black-box optimization problem where the search space is a set of
feature sets).

Model to use in wrapper-based feature selection. Available options are:

  stepup:     Start with the empty feature set, and greedily add the feature
              that most improves prediction of the target variable. Also called
              forward feature selection.

  stepdown:   Start with the full feature set, and remove the feature
              that most improves (or least decreases) prediction of the target
              variable. Also called backward / recursive feature slection or
              elimination. High computational complexity.

  none:       Do not select features using any model. Always included by
              default.

"""


WRAP_SELECT_MODEL_HELP = """
Model to use during wrapper-based (stepwise) feature selection. Available
options are:

  linear:     For classification tasks, `sklearn.linear_model.SGDClassifier`,
              and for regression tasks, `sklearn.linear_model.SGDRegressor`

  lgbm:       Use a LightGBM gradient-boosted decision tree model.

  none:       Do not select features based on any model. Always included by
              by default.

"""


SELECT_TUNE_ROUNDS_HELP = """
If not the default of 100, the number of tuning rounds to do before evaluating
each feature set during wrapper-based feature selection. Values less than 50
are probably meaningless, values over 200 are likely to be exorbitant.

"""

NORM_HELP = """
How to normalize features to [0, 1] prior to training. Available options are
`robust` and `minmax`. For short-tailed features (e.g. normally-distributed)
or uniformly distributed features, the method `robust` usually is equivalent
to minmax normalization, and no feature values are clipped, and so is the
default method.

  minmax:     Typical normalization (also sometimes called "rescaling") to
              [0, 1] by subtracting each feature minimum and then dividing by
              the feature range.

  robust:     Computes the feature "robust range" (distance from 5th to
              95th percentile, i.e. `rmin` and `rmax`, respectively, which
              yields range `rng = rmax - rmin`) and then considers values
              outside of `[rmin - 2 * rng, rmax + 2 * rng] = [xmin, xmax]` to
              be "outliers", which are then clipped (clamped) to [xmin,
              xmax]. The resulting clipped feature is then minmax normalized.

"""

NAN_HELP = """
How to handle NaN values in non-categorical features. Categorical features
are handled by representing the NaN value as another category level (class),
i.e. one extra one-hot column is created for each categorical feature with a
NaN value.

  drop:      [NOT IMPLEMENTED] Attempt to remove all non-categorical NaN
             values. Note this could remove all data if a lot of values are
             missing, which will cause errors. Currently unimplemented
             because of this, and instead defaults to `median`.

  mean:      Replace all NaN values with the feature mean value.

  median:    Replace all NaN values with the feature median value.

  impute:    Use scikit-learn experimental IterativeImputer to attempt to
             predictively fill NaN values based on other feature values. May
             be computationally demanding on larger datasets.

"""

# N_FEAT_HELP = """
# Number of features to select using method specified by --feat-select. NOTE:
# specifying values greater than e.g. 10-50 with --feat-select=step-up and slower
# algorithms can easily result in compute times of many hours.

# """

N_FEAT_NOTE = """
Only two of the three options `--n-feat-filter`, `--n-filter-cont`, and
`--n-filter-cat` may be specified at once. If all three are supplied,
`--n-feat-filter` is ignored. When only the total is supplied, it is allocated
between continuous and categorical features in proportion to their available
counts.
"""

N_FEAT_TOTAL_FILTER_HELP = f"""
Number or percentage (as a value in [0, 1]) of total features of any kind
(categorical or continuous) to select via filter-based feature selection.
{N_FEAT_NOTE}

"""

N_FEAT_WRAPPER_HELP = """
Number of features to select during wrapper-based feature selection. Note
that specifying values greater than e.g. 10-20 with slower algorithms (e.g.
LightGBM) and for data with a large number of features (e.g. over 50) can
easily result in compute times of many hours.

"""

N_FEAT_CONT_FILTER_HELP = f"""
Number or percentage (as a value in [0, 1]) of continuous features to select
via filter-based feature selection.
{N_FEAT_NOTE}

"""

N_FEAT_CAT_FILTER_HELP = f"""
Number or percentage (as a value in [0, 1]) of categorical features to select
via filter-based feature selection.
{N_FEAT_NOTE}

"""


FILTER_METHOD_HELP = """
Method(s) to use for filter selection.

Method 'assoc' is the fastest and is based on a measure of association
between the feature and the target variable, where the measure of association
is appropriate based on the cardinality (e.g. categorical vs. continuous) of
the feature and target. However, because association need not imply
generalizable predictive utility (and because the absence of an association
does not imply an absence of predictive utility), it is possible that this
method selects features that generalize poorly for prediction tasks.

Method 'pred' is based on the k-fold univariate predictive performance of
each feature on the target variable, where the estimator is a lightly tuned
sklearn.linear_model.SGDClassifier or sklearn.linear_model.SGDRegressor,
depending on the task. Computing these univariate predictive performances is
quite expensive, but because of the internal k-fold validation used, these
predictive performance metrics directly asses the potential predictive
utility of each feature in isolation.

[CURRENTLY UNIMPLEMTED] Method 'relief' is the most sophisticated and can
detect interactions among pairs of features without dramatic compute costs
(see https://www.sciencedirect.com/science/article/pii/S1532046418301400 or
https://doi.org/10.1016/j.jbi.2018.07.014 for details and overview). This is
in contrast to the 'assoc' and 'pred' methods (below) which do not detect any
feature interactions.

"""

ASSOC_SELECT_CONT_CLS_STATS = """
Type of association to use for selecting continuous features when the task or
target is classification / categorical. Options:

  t           Independent Student's t-test. Tests that means differ between
              classes.

  U           Mann-Whitney U (theoretically equivalent to AUROC below in this
              case). Assumes equal variance. Tests that medians differ between
              classes.

  W           Brunner-Munzel W. Like Mann-Whitney U, but no assumption of
              equal variance.

  corr        Pearson's correlation coefficient. Measures linear association.

  cohen_d     Cohen's D effect size. A standardized measure of the separation
              between two groups means, where the standardizer is the pooled
              variance.

  AUROC       Area under the Receiver Operating Characteristic curve. Identical
              to a scaled version of Mann-Whitney U.

  mut_info    Mutual information. See
              https://scikit-learn.org/stable/modules/generated/
              sklearn.feature_selection.mutual_info_classif.html

"""

ASSOC_SELECT_CAT_CLS_STATS = """
Type of association to use for selecting categorical features when the task or
target is classification / categorical.

  mut_info    Mutual information. See
              https://scikit-learn.org/stable/modules/generated/
              sklearn.feature_selection.mutual_info_classif.html

  H           Kruskal-Wallis H. Extension of Mann-Whitney U test to multiple
              groups, i.e. tests whether one group has a significantly more
              extreme median than the rest.

  cramer_v    Cramer's V. See https://en.wikipedia.org/w/index.php?title=
              Cram%%C3%%A9r%%27s_V&oldid=1170968029 for details. Can be loosely
              interpreted as a correlation between categorical variables.

"""

ASSOC_SELECT_CONT_REG_STATS = """
Type of association to use for selecting continuous features when the task or
target is regression / continuous.

  pearson_r   Pearson's correlation coefficient. Measures linear association.

  spearman_r  Spearman's rank correlation coefficient (Pearson's correlation,
              but on the variable ranks). Roughly, measures how strong a
              monotonic relationship is between variables.

  mut_info    Mutual information. See
              https://scikit-learn.org/stable/modules/generated/
              sklearn.feature_selection.mutual_info_regression.html

  F           F-test, technically, sklearn.feature_selection.f_regression.
              Only captures linear associations.

"""

ASSOC_SELECT_CAT_REG_STATS = """
Type of association to use for selecting categorical features when the task or
target is regression / continuous.

  mut_info    Mutual information. See
              https://scikit-learn.org/stable/modules/generated/
              sklearn.feature_selection.mutual_info_regression.html

  H           Kruskal-Wallis H. Extension of Mann-Whitney U test to multiple
              groups, i.e. tests whether one group has a significantly more
              extreme median than the rest.

"""

REG_OPTIONS = "".join(
    [f"  {f'{score.value}:': <11}{score.longname()}\n\n" for score in RegScore]
)
PRED_SELECT_REG_SCORE = f"""
Regression score to use for filter-based selection of features. Options:\n
{REG_OPTIONS}

"""

CLS_OPTIONS = "".join(
    [f"  {f'{score.value}:': <11}{score.longname()}\n\n" for score in ClsScore]
)
PRED_SELECT_CLS_SCORE = f"""
Classification score to use for filter-based selection of features. Options:\n
{CLS_OPTIONS}

"""

REDUNDANT_SELECTION = """
If this flag is present, then perform an "extra greedy" stepwise selection
that adds or removes multiple features at each step, depending on if the
scores for those features do not differ from the highest score specified by
`--redundant-threshold`.
"""

REDUNDANT_THRESHOLD = """
The threshold for deciding if wrapper feature selection scores during one
iteration of selection differ meaningfully. E.g. if the threshold is `T`, and
scoring is by accuracy, and at iteration `i` the best feature score is
`acc_max`, then all features with scores greater than or equal to `acc_max -
T` will be considered to be redundant to the first feature with an accuracy
of `acc_max`.
"""

REDUNDANT_CORR_THRESHOLD = """
During each iteration of redundant wrapper selection, while some features may
have nearly identical scores to the best score, some of these features may
nevertheless contain very different information. Feature selection is done on
the one-hot encoded categoricals and normalized continuous features. This
means correlation (e.g. Pearson) gives us a rough measure of association
between all selectable features. A more "cautious" redundant approach will
only lump in features as equivalent if they are also strongly correlated with
the best-scoring feature. This argument sets that threshold. That is, if `f`
is the feature with the best score at an iteration, and if F is the set of
features `f_i` that are within the redundancy threshold of `f`, then in this
case the set of final features R that are redundant to `f` is the set of
{{f_i | abs(corr(f, f_i)) >= T}}, i.e. features with weak (absolute magnitude
lower than threshold) correlations with `f` will remain in the pool of
features to consider in the next iteration.

"""

# HTUNE_HELP = """
# If provided, use Optuna TPESampler to attempt to optimize classifier performance
# prior to fitting and evaluating.

# """

# HTUNEVAL_HELP_STR = """
# If hyperparamater tuning using `--htune` option, specifies the validation style
# to use internally for each Optuna trial. Number of trials is specified by
# `--htune-trials`, so the number of estimator fits interacts with that values
# and `--htune-val-size`, which has a different meaning depending on validation
# type chosen. Available options:

#   holdout:    Create a single holdout set and use to validate all Optuna trials.
#               The float value in (0, 1) specified in `--htune-val-size` sets the
#               percentage of samples to use for this holdout / test set.

#   kfold:      Use k-fold cross validation to compute performance for each Optuna
#               trial. The value for `k` is specified by `--htune-val-size`.

#   loocv:      Use Leave-One-Out cross validation. `--htune-val-size` is ignored
#               in this case.

#   mc:         Use "Monte-Carlo Cross Validation", e.g. generate multiple random
#               train / test splits, i.e. equivalent to using all the splits
#               generated by `sklearn.model_selection.StratifiedShuffleSplit`
#               Currently generates 20 splits at 80%%/20%% train/test.
#               `--htune-val-size` specifies the size of the test split in this
#               case.

#   none:       Just fit the full data (e.g. validate on fitting / training data).
#               Fast but highly biased. `--htune-val-size` is ignored in this case.

# """

# HTUNE_VALSIZE_HELP = """
# See documentation for `--htune-val` (directly above if using `--help`). The
# meaning of this argument depends on the choice of `--htune-val`:

#   holdout:    A float in (0, 1) specifies the proportion of samples from the
#               input spreadsheet or table to set aside for evaluating during
#               hyperparameter tuning.

#               An integer specifies the number of samples to set aside for
#               testing.

#   kfold:      An integer being one of 3, 5, 10, or 20 specifies the number of
#               folds.

#   loocv:      Ignored.

#   mc:         A float in (0, 1) specifies the proportion of samples from the
#               input spreadsheet or table to set aside for evaluating during
#               each Monte-Carlo repeat evaluation.

#               An integer specifies the number of samples to set aside for
#               each repeat.

# """

# MC_REPEATS_HELP = """
# Ignored unless using Monte-Carlo style cross validation via `--htune-val mc`.
# Otherwise, specifies the number of random subsets of proportion
# `--htune-val-size` on which to validate the data. Default 10.

# """

HTUNE_TRIALS_HELP = """
Specifies number of trials in Optuna Study, and for each estimator and feature
selection method. E.g. fitting two estimators using three feature selection
methods with `--htune-trials=100` will result in 2 x 5 x 100 = 600 trials. If
also using e.g. the default 3-fold validation for `--htune-val-sizes`, then the
total number of estimator fits from tuning will be 600 x 3.

NOTE: if you can afford it, it is strongly recommended to set this value to a
minimum of 100 (default), or 50 if your budget is constrained. Lower values
often will fail to find good fits, given the wide range on hyperparameters
needed to make Optuna generally useful.

NOTE: Currently, df-analyze configures Optuna to use both early stopping (i.e.
do not keep testing new hyperparameters when model performance has not improved
for a number of trials) and pruning (by not performing the remaining of the five
k-fold fits and evaluations when the first 2 are clearly terrible). Thus, while
the *maximum* number of trials will be the value specified in this argument, in
practice for most models Optuna may stop early at about 50 trials or so.

df-analyze also currently configures Optuna with a *timeout* specific to each
model, so that tuning will not exceed a certain amount of time:

  mlp         60 minutes
  lgbm        60 minutes
  rf          60 minutes
  knn         30 minutes
  elastic     30 minutes
  sgd         15 minutes
  lr          15 minutes
  svm         10 minutes  [disabled]

Logistic Regression (lr) and Support Vector Machines (svm) are currently given
very little time since certain hyperparameters (namely, the regularization
parameter) can cause extremely large fit times, even on small data.

"""

CLS_TUNE_METRIC = """
Metric to use for tuning classifiers during hyperparameter tuning via Optuna.
Available options:

  acc         Accuracy

  sens        Sensitivity

  spec        Specificity

  ppv         Positive Predictive Value

  npv         Negative Predictive Value

  f1          F1 score

  bal-acc     Balanced accuracy (average of recall for each class)

Note: Option `auroc` is a valid argument, but is INVALID for tuning. AUROC
requires that the classifier output probabilities. This can be forced only by
performing classifier calibration, which requires another internal k-fold,
and is thus too expensive for the tuning process. If passing `auroc` as an
argument, a warning will be raised and `bal-acc` (balanced accuracy) will be
used instead.

"""

REG_TUNE_METRIC = """
Metric to use for tuning regressors during hyperparameter tuning via Optuna.
Available options:

  mae         Mean Absolute Error

  msqe        Mean Squared Error

  mdae        Median Absolute Error

  r2          R-squared

  var-exp     Percent variance explained

"""

# TEST_VAL_HELP = """
# Specify which validation method to use for testing. Same behavior as for
# `--htune-val` argument (see above).

# """

TEST_VALSIZES_HELP = """
Specify size of final holdout test set (not used for either tuning or feature
selection). A float in (0, 1) specifies the proportion of samples from the
input spreadsheet or table to set aside for evaluating during hyperparameter
tuning.

An integer specifies the number of samples to set aside for testing.

"""

MT_AGG_STRATEGY_HELP_STR = """
How to combine feature selection across targets. `borda` combines the
per-target ranks; `freq` favors features selected for more targets.

"""

MT_TOP_K_HELP_STR = """
After combining results across targets, keep only the top K features.
Leave this unset to keep every feature selected for at least one target.

"""

ERROR_CONSISTENCY_HELP = """
Run error consistency after feature selection and tuning. The training folds
change, but every fitted model predicts the same external holdout. Do not use a
final-test holdout to select a model.

"""

EC_FOLDS_HELP = """
Number of folds in each error-consistency repetition.

"""

EC_REPETITIONS_HELP = """
Number of shuffled K-fold repetitions. Each configuration is fitted
--ec-folds * --ec-repetitions times. The default is 5.

"""

EC_MODEL_SEED_MODE_HELP = """
Choose whether the model seed changes across folds. `vary` (the default)
includes changes from both training rows and model randomness. `fixed` uses the
same seed for every fit.

"""

EC_METHODS_HELP = """
Regression EC methods to calculate. The default is all seven methods:
ratio, ratio_diff, ratio_sign, ratio_diff_sign_magnitude,
intersection_union_sample, intersection_union_all, and
intersection_union_distance. `ratio_diff_sign_reference` is the optional signed
reference calculation and is not ranked. See docs/error_consistency.md for the
formulas.

"""

EC_HOLDOUT_ROLE_HELP = """
Tell df-analyze how the shared holdout is being used. `test` (the default)
calculates EC but disables model ranking and EC/performance correlations.
`validation` enables those outputs for a separate validation or audit set.
This option labels the existing split; it does not create a new one.

"""

EC_SAVE_PREDICTIONS_HELP = """
Save the prediction and residual/error matrices used to calculate EC, including
holdout positions and true target values.

"""

EC_OUTPUT_DETAIL_HELP = """
Choose how much EC output to keep. `summary` writes the main tables and run
details; `pairwise` adds model-pair tables and plots; `full` adds sample-level
diagnostics. --ec-save-predictions adds prediction and residual/error matrices
at any level. The default is `full`.

"""

EC_EMPTY_UNIONS_HELP = """
Handle classification pairs where neither model makes a holdout error. Choices
are 0, 1, nan, drop, error, and warn. The default, warn, issues one warning and
stores the undefined comparison as NaN.

"""

EC_EPSILON_HELP = """
Non-negative value added to regression ratio denominators near zero. The
default is 0.

"""

ADAPTIVE_ERROR_HELP = """
Estimate the chance that each holdout prediction is wrong from out-of-fold
training confidence.

"""

AER_OOF_FOLDS_HELP = """
Number of training folds used to learn the confidence-to-error mapping.

"""

AER_BINS_HELP = """
Number of confidence bins used to learn the error mapping.

"""

AER_TARGET_ERROR_HELP = """
Target error rate used when reporting risk-controlled coverage.

"""

AER_ALPHA_HELP = """
Significance level for the one-sided Clopper-Pearson error bounds, after
adjusting for the thresholds that were checked.

"""

AER_MIN_BIN_COUNT_HELP = """
Minimum number of training predictions in an adaptive-error bin.

"""

AER_PRIOR_STRENGTH_HELP = """
How strongly small bins are pulled toward the overall error rate.

"""

AER_SMOOTH_HELP = """
Disable smoothing of adaptive error bin estimates.

"""

AER_MONOTONIC_HELP = """
Require estimated error to move monotonically with confidence.

"""

AER_ADAPTIVE_BINNING_HELP = """
Use quantile-based confidence bins instead of fixed-width bins.

"""

AER_CONFIDENCE_METRIC_HELP = """
Confidence measure used for the error mapping. Use `auto` to compare the
measures available for each model.

"""

AER_NMIN_HELP = """
Minimum number of accepted holdout rows required for a reported risk threshold.

"""

AER_TOP_K_HELP = """
Analyze only the top K tuned models. The default, 0, analyzes all usable models.

"""

AER_ENSEMBLE_HELP = """
Compare supported combinations of the analyzed models.

"""

AER_ENSEMBLE_STRATEGIES_HELP = """
Ensemble strategies to run. Leave unset to use every supported strategy.

"""

AER_ENS_TOP_N_HELP = """
Number of leading models available to top-N ensemble strategies.

"""

AER_ENS_BETA_HELP = """
Beta value used to weight model confidence in adaptive-error ensembles.

"""

AER_ENS_TAU0_HELP = """
Base confidence threshold for adaptive-error ensembles.

"""

AER_ENS_LAMBDA_HELP = """
Mixing weight used by regularized adaptive-error ensembles.

"""

AER_ENS_ALPHA_HELP = """
Exponent applied to model support in adaptive-error ensemble scores.

"""

AER_ENS_TRIM_Q_HELP = """
Confidence quantile below which ensemble members are omitted.

"""

AER_ENS_TAU_LOW_HELP = """
Lower confidence cutoff for adaptive-error ensembles.

"""

AER_ENS_TAU_HIGH_HELP = """
Upper confidence cutoff for adaptive-error ensembles.

"""

OUTDIR_HELP = """
Location for results and cached computations. If omitted, df-analyze tries the
user directory, current directory, and finally a temporary directory.

"""

NO_PRED_HELP = """
Do not compute univariate predictions, and thus do not filter features based
on univariate predictive utility. Useful for datasets with over 1 million
samples and a large (e.g. 30+) number of features, where even scikit-learn
SGDClassifier and SGDRegressor are often too slow / expensive.
"""

VERBOSITY_HELP = """
Controls amount of output to stdout and stderr. Options:

  0:         ERROR: Minimal output and errors only
  1:         INFO: Logging for each Optuna trial and various interim results.
  2:         DEBUG: Currently unimplemented.

"""

EXPLODE_HELP = """
If this flag is present, silence the warnings about large increases in the
number of features due to one-hot encoding of categoricals.

"""

VERSION_HELP = """
If this flag is present, just print the df-analyze version and exit.

"""

DESC = f"""
{DF_HELP_STR}
{TARGET_HELP_STR}
{CLS_HELP_STR}
"""
