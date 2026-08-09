from pathlib import Path

VERSION = "4.2.0"

ROOT = Path(__file__).resolve().parent.parent.parent
DATADIR = ROOT / "data"
TESTDATA = DATADIR / "testing"
TEMPLATES = DATADIR / "templates"
TEST_RESULTS = TESTDATA / "__TEST_RESULTS__"
FULL_RESULTS = TESTDATA / "__FULL_RESULTS__"

# Testing
SIMPLE_XLSX = TEMPLATES / "testing/spreadsheet.xlsx"
SIMPLE_CSV = TEMPLATES / "testing/spreadsheet.csv"
SIMPLE_CSV2 = TEMPLATES / "testing/spreadsheet2.csv"
SIMPLE_ODS = TEMPLATES / "testing/spreadsheet.ods"
COMPLEX_XLSX = TEMPLATES / "testing/spreadsheet_complex.xlsx"
COMPLEX_XLSX2 = TEMPLATES / "testing/spreadsheet_complex2.xlsx"

MUSHROOM_DATA = TESTDATA / "classification/mushrooms/mushrooms.parquet"
MUSHROOM_TYPES = TESTDATA / "classification/mushrooms/types.csv"
ELDER_DATA = TESTDATA / "classification/elder/measurements.csv"
ELDER_TYPES = TESTDATA / "classification/elder/types.csv"


DATAFILE = DATADIR / "MCICFreeSurfer.mat"
DATA_JSON = DATAFILE.parent / "mcic.json"
CLEAN_JSON = DATAFILE.parent / "mcic_clean.json"
UNCORRELATED = DATADIR / "mcic_uncorrelated_cols.json"

CLASSIFIERS = ["rf", "svm", "dtree", "mlp", "bag", "dummy", "lgb"]
REGRESSORS = ["linear", "rf", "svm", "adaboost", "gboost", "mlp", "knn", "lgb"]
DIMENSION_REDUCTION = ["pca", "kpca", "umap"]
WRAPPER_METHODS = ["step-up", "step-down"]
UNIVARIATE_FILTER_METHODS = ["d", "auc", "pearson", "t-test", "u-test", "chi", "info"]
"""
info   - information gain / mutual info
chi    - chi-squared
u-test - Mann-Whitney U

See https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
"""
MULTIVARIATE_FILTER_METHODS = ["fcbf", "mrmr", "relief"]
"""
FCBF - fast correlation-based filter\n
mRMR - minimal-redundancy-maximal-relevance\n
CMIM - conditional mutual information maximization\n
relief - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6299836/
         we likely want MultiSURF (not MultiSURF*), available in
         https://github.com/EpistasisLab/scikit-rebate

See:
https://en.wikipedia.org/wiki/Feature_selection
https://en.wikipedia.org/wiki/Relief_(feature_selection)


https://www.frontiersin.org/articles/10.3389/fbinf.2022.927312/full
Another popular family of filter algorithms is the Relief-based algorithm (RBA)
family (e.g., Relief (Kira and Rendell, 1992), ReliefF (Kononenko, 1994), TURF
(Moore and White, 2007), SURF (Greene et al., 2009), SURF* (Greene et al.,
2010), MultiSURF (Urbanowicz et al., 2018a), MultiSURF* (Granizo-Mackenzie and
Moore, 2013), etc…). Relief does not exhaustively search for feature
interactions. Instead, it scores the importance of a feature according to how
well the feature’s value distinguishes samples that are similar to each other
(e.g., similar genotype) but belong to different classes (e.g., case and
control). Notably, RBAs can detect pair-wise feature interactions, some RBAs
(e.g., ReliefF, MultiSURF) can even detect higher order (>2 way) interactions
(Urbanowicz et al., 2018a).
"""
FEATURE_CLEANINGS = ["correlated", "constant", "lowinfo"]
CLEANINGS_SHORT = ["corr", "const", "info"]
HTUNE_VAL_METHODS = ["holdout", "kfold", "k-fold", "loocv", "mc", "none"]


class __Sentinel:
    pass


SENTINEL = __Sentinel()

VAL_SIZE = 0.20
SEED = 69

DEFAULT_N_STEPWISE_SELECT = 10
"""Number of features to select during stepwise selection"""
MAX_STEPWISE_SELECTION_N_FEATURES = 20
"""Number of features to warn user about feature selection problems"""
MAX_PERF_N_FEATURES = 500
"""Number of features to warn users about general performance problems"""

N_CAT_LEVEL_MIN = 20
"""
Minimum required number of samples for level of a categorical variable to be
considered useful in 5-fold analyses.

Notes
-----

For each level of categorical variable to be predictively useful, there must
be enough samples to be statistically meaningful (or to allow some reasonable
generalization) in each fold used for fitting or analyses. Roughly, this
means each fold needs to see at least 10-20 samples of each level (depending
on how strongly / cleanly the level relates to other features - ultimately
this is just a heuristic). Assuming k-fold is used for validation, then this
means about (1 - 1/k) times 10-20 samples per categorical level would a
reasonable default minimum requirement one might use for culling categorical
levels. Under the typical assumption of k=5, this means we require useful /
reliable categorical levels to have 8-16 samples each.

Inflation is to categorical variables as noise is to continuous ones.
"""

# N_TARG_LEVEL_MIN = 30
N_TARG_LEVEL_MIN = 20
"""
Minimum required number of samples for level of a categorical target variable
to be considered useful in 5-fold analyses.

Notes
-----
Suppose the smallest target class or level has N samples. With stratified
k-fold, a k-fold training set will have floor((1 - 1/k) * N) samples of that
level. For k in {3, 5}, this is either floor(2N/3) or floor(0.8*N) samples.
Now if we use k-fold again internally (e.g. nested k-fold), with internal
k=m, then there are again floor( (1 - 1/m)(1 - 1/k) * N ) samples of that
level, i.e. for (k, m) in {(3, 3), (3, 5), (5, 5)} we have floor(0.444 * N),
floor(0.5333 * N), or floor(0.64*N) = floor(16N/25) samples in the training
set. In the test set the numbers are just floor(N/9), floor(N/15),
floor(N/25).

So far we use only 5-fold, so to guarantee one sample of each level in an
internal fold, we need floor(N/25) > 1 ==> N > 25.
"""

N_MULTITARGET_LEVEL_MIN = N_TARG_LEVEL_MIN
"""
Minimum required number of samples for level of a categorical target variable
when running multi-target classification.

Notes
-----
Kept separate from N_TARG_LEVEL_MIN so multi-target filtering heuristics and
split warnings can be adjusted without changing single-target behavior.
"""

MULTITARGET_MAX_RARE_LEVEL_FRAC = 1.0 / 3.0
"""
Maximum fraction of multi-target dimensions in a row that may belong to
undersampled target levels before that row is dropped.

Notes
-----
Using one third means, for example, that a 3-target problem can retain rows
with up to one rare label while still discarding rows dominated by rare labels.
"""

N_TARG_LEVEL_MIN_INTERNAL = N_TARG_LEVEL_MIN // 2
"""
Minimum required number of samples for level of a categorical target variable
to be considered useful in internal 5-fold analyses.

Notes
-----
Consider a binary classification problem with 20 samples per target level,
i.e. src._constants.N_TARG_LEVEL_MIN condition (above in source code) is
satisfied. After setting aside a holdout set of 50%, we have 10 samples per
target level.

With 5-fold, this means each train/test split has 8 train samples, and 2 test
samples per target level, or, in total we train on 16 samples, test on 4
samples, in the binary classification case. It is extremely dubious whether
this procedure is statistically valid at all, since even under very broad
assumptions in a priori power analysis, you generally are going to need 50-60
samples (per cell or group) in order to have even a basic level of trust in
even very simple and constrained linear modeling contexts, e.g. classical
ANOVA and the like [1]:

It is difficult to translate power analysis to the k-fold validation
paradigm. However, since even highly-constrained linear models do not really
produce reliable estimates at below about 50 samples WHEN USING ALL SAMPLES,
it seems quite unlikely that 5 small samples averaged togther are going to
overcome this basic problem with noisy performance estimates.

There is a long history of debate (especially in psycholgical methods
following the replication crisis in around 2015) about whether it is better
to do e.g. one large study of N*k samples, or k small studies of N samples.
Intuitively, if this kind of thing matters, either you have a heterogeneity
problem, or you really just don't have enough data or your experiment is
poorly designed, because, regardless of the math, deciding between these two
cases should not result in huge differences. That is, if *in the simple case
of a single study* power analysis tells use we need a minimum sample size of
50 to detect anything of significance, then we should not have much
confidence in some hokey method that is able to magically claim getting
better power / significance by some funky splitting that makes use of 5
10-sample partitions, or e.g. 5 k-fold partitions of 40/10 non-overlapping
train-test splits. You still have only 50 samples, and you know that is not
generally trustworthy in the cleanest case. It is the usual dictum of: if you
need very specific statistics / assumptions to find your effect, you
*probably* don't have much of a generalizable effect.

By this reasoning, it is probably enough to use the single-study a prior power
analysis case as a general guideline for what, roughly, is an acceptable
bare minimum sample size even in fairly favorable conditions.

While I would put this at more like 50 samples per cell / group, and the old
heuristic of 20 is most certaintly widely regarded as deeply inadequate, we
can be lenient and pretend like 20 is somehow okay.

Since this is what we use for N_TARG_LEVEL_MIN, and since a default
df-analyze holdout is about 50% (under the assumption it will generally be
used on small data), we set the minimum number of sampes per target level to
be half the base requirement, i.e. 10 (even though this is really still
unacceptably low, most likely).

[1] Lakens, D. (2022). Sample Size Justification. Collabra: Psychology, 8(1),
33267.doi:10.1525/collabra.33267 https://doi.org/10.1525/collabra.33267
https://online.ucpress.edu/collabra/article/8/1/33267/120491/Sample-Size-Justification


"""

N_TARG_LEVEL_MIN_TRAIN_INTERNAL = 8
"""
As per code documentation above, we have required 10 samples per target level
in internal folds. However, test folds are of course smaller than train folds,
and for a binary classification problem, the limit of 10 means folds have
8/2 train/test samples per target level, or 4 test samples total in each of the
5 folds.

To be consistent with the reasoning for N_TARG_LEVEL_MIN_INTERNAL, we set this
value to 8, but this should really be higher.
"""

N_TARG_LEVEL_MIN_TEST_INTERNAL = 2
"""
As per code documentation above, we have required 10 samples per target level
in internal folds. However, test folds are of course smaller than train folds,
and for a binary classification problem, the limit of 10 means folds have
8/2 train/test samples per target level, or 4 test samples total in each of the
5 folds.

To be consistent with the reasoning for N_TARG_LEVEL_MIN_INTERNAL, we set this
value to 2, but this should really be higher.
"""

UNIVARIATE_PRED_MAX_N_SAMPLES = 1500
"""Maximum number of samples to use in univariate predictive analyses"""

# from https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
NAN_STRINGS = [
    "",
    "-1.#IND",
    "-1.#QNAN",
    "-nan",
    "-NaN",
    "#N/A N/A",
    "#N/A",
    "#NA",
    "<NA>",
    "1.#IND",
    "1.#QNAN",
    "n/a",
    "N/A",
    "NA",
    "nan",
    "NaN",
    "Nan",
    "None",
    "null",
    "NULL",
    "-",
    "_",
]

N_EMBED_DEFAULT = 20
N_WRAPPER_DEFAULT = 20
N_FILTER_CONT_DEFAULT = 20
N_FILTER_CAT_DEFAULT = 10
N_FILTER_TOTAL_DEFAULT = 30
P_FILTER_CONT_DEFAULT = 0.50
P_FILTER_CAT_DEFAULT = 0.50
P_FILTER_TOTAL_DEFAULT = 0.5

N_FEAT_DOWNSAMPLE_DEFAULT = 1000
DOWNSAMPLE_CHUNK_SIZE_DEFAULT = 25_000
DOWNSAMPLE_SCORE_LIMIT = 500
LARGE_FEATURE_THRESHOLD = 100_000
EXTREME_FEATURE_THRESHOLD = 1_000_000
DIRECT_DOWNSAMPLE_MAX_FEATURES = 50_000
DOWNSAMPLE_MAX_CHUNK_BYTES = 512 * 1024**2
DOWNSAMPLE_MAX_DENSE_BYTES = 2 * 1024**3
DOWNSAMPLE_DENSE_PEAK_FACTOR = 4
