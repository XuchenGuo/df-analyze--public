```text
usage:

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

options:
  -h, --help            show this help message and exit
  --spreadsheet SPREADSHEET

                        The path to the formatted spreadsheet to analyze.

                        Currently only spreadsheets saved as either `.xlsx` or `.csv` are supported.

                        If your data is saved as a Pandas `DataFrame`, it must have shape
                        `(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
                        column holding the target variable (or feature) can be specified by the
                        `--target` / `-y` argument, but is "target" by default if such a column name
                        exists, or the last column if it does not.

                        For more details, see the README at https://github.com/stfxecutables/df-analyze.

  --df DF
                        The dataframe to analyze.

                        Currently only tables saved as either `.xlsx`, `.json` or `.csv`, or NumPy
                        `ndarray`s saved as "<filename>.npy" are supported, but a file exported by
                        Pandas `DataFrame.to_*` method is preferred.

                        If your data is saved as a Pandas `DataFrame`, it must have shape
                        `(n_samples, n_features)` or `(n_samples, n_features + 1)`. The name of the
                        column holding the target variable (or feature) can be specified by the
                        `--target` / `-y` argument, but is "target" by default if such a column name
                        exists, or the last column if it does not.

  --separator SEPARATOR

                        Separator used in .csv files. Default ",".

  --target TARGET [TARGET ...]

                        The (string) name of the target variable for either regression or
                        classification.

  --targets TARGETS

                        Comma-separated target columns for a multi-target classification or
                        regression run. This takes precedence over `--target`.

  --grouper GROUPER [GROUPER ...]

                        The (string) name of the grouping variable (if one is present) which will be
                        used to ensure samples within the same group do not end up in both train and
                        test splits. I.e. the name of the feature that will be passed into scikit-learn
                        GroupStratifiedKFold.

  --categoricals CATEGORICALS

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

  --ordinals ORDINALS
                        A string or list of strings, e.g.

                            --ordinals star_rating,number_of_purchases,times_signed_in

                        that specifies which features will be treated as ordinal regardless of the
                        number of levels or format of the data. If during data cleaning ordinal
                        variables are detected that are NOT specified by the user, a warning will be
                        raised. If the values of the specified variables cannot be interpreted as
                        integers, then df-analyze will exit with an error.

                        See documentation for `--categoricals` re: the use of commas to list features.

  --drops DROPS
                        A string or list of strings, e.g.

                            --drops subject_id,location_id,useless_feature1,useless_feat2

                        that specifies which features will be removed from the data and not considered
                        for any inspection, description or univariate analysis, and which will not be
                        included in any feature selection, model tuning, or final predictive models.

                        See documentation for `--categoricals` re: the use of commas to list features.

  --mode {classify,regress}

                        If "classify", do classification. If "regress", do regression.

  --device {auto,cpu,cuda}
                        Runtime device for supported models. Auto uses workload-aware
                        thresholds for KNN, CatBoost, and XGBoost so small jobs stay on
                        CPU; compute-heavy neural and embedding backends prefer an
                        available accelerator. CUDA falls back to CPU with a warning
                        when a backend is unavailable. GANDALF may use MPS in auto mode
                        on supported Apple systems.

  --device-install {auto,ask,never}
                        Managed CUDA PyTorch setup policy. The default is never.
                        Managed setup is available from a source checkout containing
                        pyproject.toml and uv.lock.

  --tabpfn-version {v3,v2.6,v2.5}
                        TabPFN checkpoint version. The default is v3; v2.6 and v2.5
                        select older supported checkpoints. The first run requires
                        accepting the corresponding Prior Labs license and setting
                        TABPFN_TOKEN in the same terminal.

  --classifiers  [ ...]

                        The list of classifiers to use when comparing classification performance.
                        Registered tokens: [catboost dummy dtree et gandalf kan knn lgbm lr mlp rf
                        sgd svm tabpfn xgb]. The svm token is currently disabled by the CLI.

                          catboost    CatBoost classifier.

                          xgb         XGBoost classifier.

                          tabpfn      TabPFN classifier.

                          dtree       scikit-learn DecisionTreeClassifier.

                          et          scikit-learn ExtraTreesClassifier.

                          knn         scikit-learn KNeighborsClassifier.

                          lgbm        LightGBM boosted decision tree classifier.

                          rf          LightGBM random forest classifier.

                          lr          scikit-learn LogisticRegression.

                          sgd         scikit-learn SGDClassifier.

                          mlp         Modern multi-layer perceptron implemented in skorch/PyTorch.

                          kan         Kolmogorov-Arnold network implemented with PyKAN/skorch.

                          svm         Registered for compatibility; currently disabled.

                          gandalf     Gated Adaptive Network for Deep Automated Learning of
                                      Features for Tabular Data: https://arxiv.org/abs/2207.08548

                          dummy       scikit-learn DummyClassifier.

  --regressors  [ ...]

                        The list of regressors to use when comparing regression model performance.
                        Registered tokens: [catboost dummy dtree elastic et gandalf kan knn lgbm mlp
                        rf sgd svm tabpfn xgb]. The svm token is currently disabled by the CLI.

                          catboost    CatBoost regressor.

                          xgb         XGBoost regressor.

                          tabpfn      TabPFN regressor.

                          dtree       scikit-learn DecisionTreeRegressor.

                          et          scikit-learn ExtraTreesRegressor.

                          knn         scikit-learn KNeighborsRegressor.

                          lgbm        LightGBM boosted decision tree regressor.

                          rf          LightGBM random forest regressor.

                          elastic     scikit-learn ElasticNet.

                          sgd         scikit-learn SGDRegressor.

                          mlp         Modern multi-layer perceptron implemented in skorch/PyTorch.

                          kan         Kolmogorov-Arnold network implemented with PyKAN/skorch.

                          svm         Registered for compatibility; currently disabled.

                          gandalf     Gated Adaptive Network for Deep Automated Learning of
                                      Features for Tabular Data: https://arxiv.org/abs/2207.08548

                          dummy       scikit-learn DummyRegressor.

  --feat-select  [ ...]

                        The feature selection method(s) to use. Available options are:

                          filter      Select features based on their univariate relationships to the
                                      target variables.

                          embed       Select features using a model with implicit feature selection,
                                      e.g. an L1-regularized model or decision tree. For avaialable
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

  --feat-downsample {none,auto,random,variance,f-test,mutual-info,linear,lgbm,svd,sparse-rp,rank-ensemble,selector-ensemble,stable-rank}

                        Reduce a wide matrix before the usual feature-selection and tuning
                        stages. Supervised methods use a screening subset disjoint from the
                        rows used for model tuning. The default is none.

  --n-feat-downsample N_FEAT_DOWNSAMPLE

                        Maximum number or fraction of features retained.

  --downsample-variance-threshold DOWNSAMPLE_VARIANCE_THRESHOLD

                        Minimum sample variance retained by variance downsampling.

  --downsample-save-scores

                        Save scores for every feature instead of only the leading scores.

  --downsample-chunk-size DOWNSAMPLE_CHUNK_SIZE

                        Maximum number of source features scored in one chunk.

  --downsample-screening-fraction DOWNSAMPLE_SCREENING_FRACTION

                        Fraction of training rows used only for supervised downsampling.

  --large-feature-mode

                        Split and downsample a numeric table before normal preparation.

  --assume-numeric-features

                        Skip the pandas dtype check in large-feature table mode; all predictor values
                        must still convert to finite numbers.

  --skip-full-prepared-save-before-downsample

                        Do not save the full prepared matrix before downsampling.

  --input-format {auto,table,svmlight}

                        Input format; auto recognizes common SVMlight suffixes.

  --svmlight-index-base {auto,zero,one}

                        Feature index base used to name SVMlight columns.

  --mt-agg-strategy {borda,freq}

                        How to combine per-target feature-selection results. Borda combines
                        ranks; freq emphasizes features selected for several targets.

  --mt-top-k MT_TOP_K

                        Keep only the top K features after multi-target aggregation. When
                        omitted, all features that pass the aggregation criteria are retained.

  --embed-select  [ ...]

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

  --wrapper-select
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

  --wrapper-model
                        Model to use during wrapper-based (stepwise) feature selection. Available
                        options are:

                          linear:     For classification tasks, `sklearn.linear_model.SGDClassifier`,
                                      and for regression tasks, `sklearn.linear_model.SGDRegressor`

                          lgbm:       Use a LightGBM gradient-boosted decision tree model.

                          none:       Do not select features based on any model. Always included by
                                      by default.

  --norm {minmax,robust}

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

  --nan {drop,mean,median,impute}

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

  --n-feat-filter N_FEAT_FILTER

                        Number or percentage (as a value in [0, 1]) of total features of any kind
                        (categorical or continuous) to select via filter-based feature selection.

                        Note only two of two of the three options:`--n-feat-filter`,
                        `--n-filter-cont`, and  `--n-filter-cat` may be specified at once,
                        otherwise the `--n-feat-filter` argument will be ignored.


  --n-feat-wrapper N_FEAT_WRAPPER

                        Number of features to select during wrapper-based feature selection. Note
                        that specifying values greater than e.g. 10-20 with slower algorithms (e.g.
                        LightGBM) and for data with a large number of features (e.g. over 50) can
                        easily result in compute times of many hours.

  --n-filter-cont N_FILTER_CONT

                        Number or percentage (as a value in [0, 1]) of continuous features to select
                        via filter-based feature selection.

                        Note only two of two of the three options:`--n-feat-filter`,
                        `--n-filter-cont`, and  `--n-filter-cat` may be specified at once,
                        otherwise the `--n-feat-filter` argument will be ignored.


  --n-filter-cat N_FILTER_CAT

                        Number or percentage (as a value in [0, 1]) of categorical features to select
                        via filter-based feature selection.

                        Note only two of two of the three options:`--n-feat-filter`,
                        `--n-filter-cont`, and  `--n-filter-cat` may be specified at once,
                        otherwise the `--n-feat-filter` argument will be ignored.


  --filter-method
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

  --filter-assoc-cont-classify

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

  --filter-assoc-cat-classify

                        Type of association to use for selecting categorical features when the task or
                        target is classification / categorical.

                          mut_info    Mutual information. See
                                      https://scikit-learn.org/stable/modules/generated/
                                      sklearn.feature_selection.mutual_info_classif.html

                          H           Kruskal-Wallace H. Extension of Mann-Whitney U test to multiple
                                      groups, i.e. tests whether one group has a significantly more
                                      extreme median than the rest.

                          cramer_v    Cramer's V. See https://en.wikipedia.org/w/index.php?title=
                                      Cram%C3%A9r%27s_V&oldid=1170968029 for details. Can be loosely
                                      interpreted as a correlation between categorical variables.

  --filter-assoc-cont-regress

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

  --filter-assoc-cat-regress

                        Type of association to use for selecting categorical features when the task or
                        target is regression / continuous.

                          mut_info    Mutual information. See
                                      https://scikit-learn.org/stable/modules/generated/
                                      sklearn.feature_selection.mutual_info_regression.html

                          H           Kruskal-Wallace H. Extension of Mann-Whitney U test to multiple
                                      groups, i.e. tests whether one group has a significantly more
                                      extreme median than the rest.

  --filter-pred-regress {mae,msqe,mdae,r2,var-exp}

                        Regression score to use for filter-based selection of features. Options:

                          mae:       Mean Absolute Error

                          msqe:      Mean Squared Error

                          mdae:      Median Absolute Error

                          r2:        R-Squared

                          var-exp:   Percent Variance Explained



  --filter-pred-classify {acc,auroc,sens,spec}

                        Classification score to use for filter-based selection of features. Options:

                          acc:       Accuracy

                          auroc:     Area Under the ROC Curve

                          sens:      Sensitivity

                          spec:      Specificity



  --redundant-wrapper-selection

                        If this flag is present, then perform an "extra greedy" stepwise selection
                        that adds or removes multiple features at each step, depending on if the
                        scores for those features do not differ from the highest score specified by
                        `--redundant-threshold`.
  --redundant-threshold REDUNDANT_THRESHOLD

                        The threshold for deciding if wrapper feature selection scores during one
                        iteration of selection differ meaningfully. E.g. if the threshold is `T`, and
                        scoring is by accuracy, and at iteration `i` the best feature score is
                        `acc_max`, then all features with scores greater than or equal to `acc_max -
                        T` will be considered to be redundant to the first feature with an accuracy
                        of `acc_max`.
  --redundant-corr-threshold REDUNDANT_CORR_THRESHOLD

                        During each iteration of redundant wrapper selection, while some features may
                        have nearly identical scores to the best score, some of these features may
                        nevertheless contain very different information. Feature selection is done on
                        the one-hot encoded cateogoricals and normalized continuous features. This
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

  --htune-trials HTUNE_TRIALS

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

  --htune-cls-metric {acc,auroc,sens,spec,ppv,npv,f1,bal-acc}

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

  --htune-reg-metric {mae,msqe,mdae,r2,var-exp}

                        Metric to use for tuning regressors during hyperparameter tuning via Optuna.
                        Available options:

                          mae         Mean Absolute Error

                          msqe        Mean Squared Error

                          mdae        Median Absolute Error

                          r2          R-squared

                          var-exp     Percent variance explained

  --test-val-size TEST_VAL_SIZE

                        Specify size of final holdout test set (not used for either tuning or feature
                        selection). A float in (0, 1) specifies the proportion of samples from the
                        input spreadsheet or table to set aside for evaluating during hyperparameter
                        tuning.

                        An integer specifies the number of samples to set aside for testing.

  --error-consistency, --ec
                        Enable repeated K-fold error consistency analysis on the final
                        holdout set.
  --ec-folds EC_FOLDS
                        Number of folds in each error-consistency repetition (minimum 2).
  --ec-repetitions EC_REPETITIONS
                        Number of independently shuffled K-fold repetitions (minimum 1,
                        default 5).
  --ec-model-seed-mode {vary,fixed}
                        Whether fitted-model RNG seeds vary deterministically by fold
                        (split plus algorithmic instability) or stay fixed (training-
                        subset sensitivity). Default: vary.
  --ec-methods EC_METHODS [EC_METHODS ...]
                        Regression EC metrics to compute. The default is all seven methods.
  --ec-save-predictions
                        Save the holdout prediction and residual/error matrices used by EC.
  --ec-empty-unions {0,1,nan,drop,error,warn}
                        Classification policy when neither model in a pair makes an error.
                        The default is warn, which records the undefined comparison as NaN.
  --ec-epsilon EC_EPSILON
                        Non-negative denominator stabilization for regression ratio metrics.
  --ec-recurrence-threshold EC_RECURRENCE_THRESHOLD
                        Error-rate threshold used to label high recurrence in the joint
                        adaptive-error and error-consistency report. This is a
                        configurable heuristic, not a universal scientific cutoff.
                        Default: 0.5.

  --outdir OUTDIR
                        Specifies location of all results, as well as cache files for slow
                        computations (e.g. stepwise feature selection). If unspecified, will attempt
                        to default to a number of common locations (/Users/derekberger, the
                        current working directory /Users/derekberger/Documents/Antigonish/df-analyze, or a temporary directory).

  --no-preds
                        Do not compute univariate predictions, and thus do not filter features based
                        on univariate predictive utility. Useful for datasets with over 1 million
                        samples and a large (e.g. 30+) number of features, where even scikit-learn
                        SGDClassifer and SGDRegressor are often too slow / expensive.
  --verbosity VERBOSITY

                        Controls amount of output to stdout and stderr. Options:

                          0:         ERROR: Minimal output and errors only
                          1:         INFO: Logging for each Optuna trial and various interim results.
                          2:         DEBUG: Currently unimplemented.

  --no-warn-explosion
                        If this flag is present, silence the warnings about large increases in the
                        number of features due to one-hot encoding of categoricals.

  --version
                        If this flag is present, just print the df-analyze version and exit.


USAGE EXAMPLE (assumes you have run `poetry shell`):

    python df-analyze.py \
        --df weather_data.json \
        --target temperature \
        --categoricals weekday,season \
        --ordinals rainfall_mm \
        --drops date,sample_id \
        --mode=regress \
        --regressors=elastic lgbm knn \
        --feat-select wrap embed filter \
        --embed-select linear \
        --wrapper-select step-up \
        --wrapper-model linear \
        --norm robust \
        --nan median \
        --n-feat-filter 10 \
        --n-feat-wrappper 10 \
        --test-val-size=0.25 \
        --outdir='./results'
```
