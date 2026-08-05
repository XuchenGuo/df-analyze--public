[![DOI](https://zenodo.org/badge/364694785.svg)](https://zenodo.org/badge/latestdoi/364694785)

<!-- omit from toc -->
# Contents

- [Overview](#overview)
  - [For Students or Novices to Machine and Deep Learning](#for-students-or-novices-to-machine-and-deep-learning)
- [Installation](#installation)
  - [Installation via `uv`](#installation-via-uv)
  - [\[\*\***LEGACY**\*\*\] Local Install by Shell Script](#legacy-local-install-by-shell-script)
  - [By Singularity / Apptainer Container](#by-singularity--apptainer-container)
  - [Windows Support](#windows-support)
- [Usage](#usage)
  - [CPU and CUDA Devices](#cpu-and-cuda-devices)
  - [Quick Start and Examples](#quick-start-and-examples)
    - [Using Builtin Data](#using-builtin-data)
    - [Additional Model Backends](#additional-model-backends)
  - [Multi-Target Analysis](#multi-target-analysis)
  - [Using a `df-analyze`-formatted Spreadsheet](#using-a-df-analyze-formatted-spreadsheet)
    - [Overriding Spreadsheet Options](#overriding-spreadsheet-options)
  - [Embedding Functionality](#embedding-functionality)
    - [Quickstart](#quickstart)
    - [About the Embedding Models](#about-the-embedding-models)
    - [Supported Dataset Formats](#supported-dataset-formats)
      - [Image Data](#image-data)
      - [Text Data](#text-data)
  - [Large-Scale Feature Downsampling](#large-scale-feature-downsampling)
  - [Usage on Compute Canada / Digital Research Alliance of Canada / Slurm HPC Clusters](#usage-on-compute-canada--digital-research-alliance-of-canada--slurm-hpc-clusters)
    - [Building the Singularity Container](#building-the-singularity-container)
    - [Using the Singularity Container](#using-the-singularity-container)
- [Analysis Pipeline](#analysis-pipeline)
    - [Feature Type and Cardinality Inference](#feature-type-and-cardinality-inference)
    - [Data Preparation](#data-preparation)
      - [Categorical Deflation](#categorical-deflation)
        - [Categorical Target Deflation](#categorical-target-deflation)
    - [Data Splitting](#data-splitting)
    - [Univariate Feature Analyses](#univariate-feature-analyses)
    - [Feature Selection](#feature-selection)
      - [Redundancy-Aware Feature Selection \*\*\[NEW\]\*\*](#redundancy-aware-feature-selection-new)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Final Validation](#final-validation)
    - [Adaptive Error Analysis](#adaptive-error-analysis)
    - [Error Consistency](#error-consistency)
- [Program Outputs](#program-outputs)
  - [Order of Reading](#order-of-reading)
  - [Subdirectories](#subdirectories)
    - [📂 Hashed Subdirectories](#-hashed-subdirectories)
    - [`📂 inspection`](#-inspection)
      - [Destructive Data Changes](#destructive-data-changes)
    - [`📂 prepared`](#-prepared)
    - [`📂 features`](#-features)
      - [`📂 downsampling`](#-downsampling)
      - [`📂 associations`](#-associations)
      - [`📂 descriptions`](#-descriptions)
      - [`📂 predictions`](#-predictions)
    - [`📂 selection`](#-selection)
      - [`📂 embed`](#-embed)
      - [`📂 filter`](#-filter)
      - [`📂 wrapper`](#-wrapper)
    - [`📂 tuning`](#-tuning)
    - [`📂 results`](#-results)
- [Complete Listing](#complete-listing)
- [Limitations](#limitations)
  - [Target Variables](#target-variables)
  - [Dataset Size](#dataset-size)
  - [Inappropriate Data](#inappropriate-data)
  - [Inappropriate Tasks](#inappropriate-tasks)
- [Currently Implemented Program Features and Analyses](#currently-implemented-program-features-and-analyses)
  - [Completed Features](#completed-features)
    - [Automated Data Preprocessing](#automated-data-preprocessing)
    - [Feature Descriptive Statistics](#feature-descriptive-statistics)
    - [Univariate Feature-Target Associations](#univariate-feature-target-associations)
    - [Univariate Prediction Metrics for each Feature-Target Pair](#univariate-prediction-metrics-for-each-feature-target-pair)


# Overview

`df-analyze` is a command-line tool for performing
[AutoML](https://en.wikipedia.org/w/index.php?title=Automated_machine_learning&oldid=1193286380)
on small to medium-sized tabular datasets. Separate downsampling paths are
available for much wider numeric data, including sparse SVMlight input.
`df-analyze` attempts to automate:

- feature type inference
- feature description (e.g. univariate associations and stats)
- data cleaning (e.g. NaN handling and imputation)
- training, validation, and test splitting
- feature selection
- optional large-scale feature downsampling for very wide numeric or SVMlight data
- hyperparameter tuning
- model selection and validation
- single- and multi-target classification or regression
- CPU/CUDA execution support with automatic per-component device routing
- optional adaptive-error and repeated K-fold error-consistency diagnostics

and saves all key tables and outputs from this process.

See [Large-scale feature downsampling](docs/feature_downsampling.md) for the
chunked, leakage-controlled preprocessing options for wide feature matrices.

**\*\*UPDATE - September 30 2024\*\*** Now, `df-analyze` supports [zero-shot
embedding](#embedding-functionality) of image and text data via the
`df-embed.py` script. This allows the conversion of [correctly
formatted](#supported-dataset-formats) image and text datasets into tabular
data that can be handled by the standard `df-analyze` tabular prediction
tools.

Currently, significant efforts have been made to make `df-analyze` robust to
a wide variety of tabular datasets. However, there are some significant
[limitations](#limitations).

If you have any questions about:

- how to get `df-analyze` installed and working on your machine
- understanding `df-analyze` outputs
- making good choices about `df-analyze` options for your dataset
- any problems you might run into using `df-analyze`
- general issues with the approach and/or code of `df-analyze`
- contributing to `df-analyze`
- anything else that you think is relevant to the `df-analyze` software
  specifically (and not course-related issue or complaints, if encountering
  `df-analyze` as part of a [university/college
  course](#for-students-or-novices-to-machine-and-deep-learning))

Don't be shy about asking for help in the
[Discussions](https://github.com/stfxecutables/df-analyze/discussions)!


## For Students or Novices to Machine and Deep Learning

For students encountering `df-analyze` through a course, see the [student
README](docs/students.md)
[WIP!] in this repo. The student README contains some descriptions and tips
that are helpful for those just starting to learn about the CLI, containers,
AutoML tools, and also some explanations and tips for running `df-analyze` on
SLURM High-Performance Computing (HPC) clusters, particularly the [Digital
Research Alliance of Canada (formerly Compute Canada)
clusters](https://docs.alliancecan.ca/wiki/Technical_documentation).



# Installation

Currently, `df-analyze` is distributed as Python scripts dependent on the
contents of this repository. So, to run `df-analyze`, you will generally have
to clone this repository and install a compatible virtual environment. This
can now be done easily and quickly by [`uv`](https://docs.astral.sh/uv/)
([instructions](#installation-via-uv)), but if for some reason you cannot use
`uv`, there is the legacy fallback [`local_install.sh` shell install
script](#local-install-by-shell-script).

**Note**: If you are an advanced user working on an HPC cluster (e.g. Compute
Canada / DRAC), you may need to [build a container to make use of
`df-analyze` reliably](#building-the-singularity-container).


## Installation via `uv`

1. Install `uv` as per the [installation
   instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
   - i.e. if you are on macOS or Linux: `curl -LsSf
   https://astral.sh/uv/install.sh | sh` - or if you are on Windows:
   `powershell -ExecutionPolicy ByPass -c "irm
   https://astral.sh/uv/install.ps1 | iex"`
2. Make sure you have [installed Git](https://git-scm.com/install/). If you
   are on macOS or Linux, this should be there by default and this step is
   not required.
   - Windows users looking for guidance during installation of Git should see
     the [Windows install instructions in this
     repo](docs/windows_install.md)
3. Navigate to a suitable directory, e.g. `~/Documents` and clone the
   repository:
   ```shell
   cd ~/Documents
   git clone https://github.com/stfxecutables/df-analyze.git
   ```
4. Run `uv sync`

Now df-analyze is installed, and you can run e.g.

```shell
uv run python df-analyze.py --version
```

to see the version number, or e.g.

```shell
uv run python df-analyze.py --help
```

to see the complete command-line documentation.



**Note**: If you are on Windows you may get errors with the above procedure.
If the error involves `import torch` and a function `_load_dll_libraries()`,
with an error message of the form:

> Microsoft Visual C++ Redistributable is not installed, this may lead to the
> DLL load failure. It can be downloaded at [platform-specific link]

try installing the missing DLL from the link provided. Otherwise, try the
[Windows install
instructions](docs/windows_install.md).
If you see any issues during installation, feel free to reach out
in the
[Discussions](https://github.com/stfxecutables/df-analyze/discussions), file
an [Issue](https://github.com/stfxecutables/df-analyze/issues), or try the
[legacy installation procedure](#local-install-by-shell-script).



## [\*\***LEGACY**\*\*] Local Install by Shell Script

After having cloned the repo, the
[`local_install.sh`](local_install.sh)
script can be used to install the dependencies for `df-analyze`. You will
need to first install [`pyenv`](https://github.com/pyenv/pyenv) (or
[`pyenv-win`](https://github.com/pyenv-win/pyenv-win) on Windows) in order
for the install script to work, but then the script will compile Python
3.12.5 and create a virtual environment with the necessary dependencies when
you run

```sh
bash local_install.sh
```

Then, you can activate the installed virtual environment by running

```sh
source .venv/bin/activate
```

in your shell (or by running `.venv/scripts/activate` on Windows -  see also
[here](https://virtualenv.pypa.io/en/legacy/userguide.html#activate-script)
if you run into permission issues when doing this). You should be able to see
if the install working by running:

```sh
python df-analyze.py --help

```

This install procedure *should* work on macOS (including Apple Silicon, e.g.
MX series macs), and on most major and up-to-date Linux distributions, and on
Windows in the Windows Subsystem for Linux (WSL). However, Windows users
wishing to avoid using the WSL should adapt the [install
script](local_install.sh)
for their needs.


<!-- ## Quickstart With `pip`

To install, run:

```sh
pip install df-analyze
```

This should "just work", though this may not install the latest version. Then,
to see the available options, just run:

```sh
df-analyze --help
```

to see usage and available options. -->



## By Singularity / Apptainer Container

Alternately, [build the Singularity / Apptainer
container](#building-the-singularity-container) and use this for running any
code that uses `df-analyze`. This should work on any Linux system (including
HPC systems / clusters like Compute Canada / DRAC).


## Windows Support

Native Windows support is still experimental. The automated tests cover the
Windows installation helpers and long output paths, but not every optional
model and GPU configuration. The Windows Subsystem for Linux (WSL) generally
works well, and the [local install scripts](#legacy-local-install-by-shell-script)
should work there.

If for some reason you can't use the WSL, then there are experimental manual
Windows installation instructions
[here](docs/windows_install.md).

# Usage

For full documentation of `df-analyze` run:

```shell
uv run python df-analyze.py --help
```

or

```shell
python df-analyze.py --help
```

if you are using the legacy `pyenv` install with an activated virtual
environment. **Note**: all future command line examples will omit the `uv run`
command, with the assumption that this is implicit.

Alternately, you can see what the `--help` option outputs
[here](docs/arguments.md),
but keep in mind the actual outputs of the `--help` command are less likely to
be out of date.

For documentation of the embedding functionality, run:

```shell
df-embed --help
```

The equivalent command from a source checkout is:

```shell
python df-embed.py --help
```


## CPU and CUDA Devices

Both `df-analyze` and `df-embed` accept `--device auto`, `--device cpu`, and
`--device cuda`. This option controls the parts of the run that support a GPU;
preprocessing and CPU-only models still run on the CPU.

- `auto` is the recommended default. KNN, CatBoost, and XGBoost use workload
  thresholds; neural models, TabPFN, and embeddings use CUDA when it is
  available. If an `auto` task encounters a CUDA runtime error, the complete
  affected task is tried once more on the CPU.
- `cpu` runs everything on the CPU and does not check for a GPU.
- `cuda` requires CUDA for selected models that support it. A missing backend
  or CUDA runtime failure stops the run instead of falling back. CPU-only
  models and preprocessing still run on the CPU. The command also stops if
  none of the selected work has a supported CUDA backend.

CatBoost, XGBoost, KNN, MLP, KAN, GANDALF, TabPFN, image and text embedding,
and error-consistency calculations can use CUDA. Neural models and embedding,
together with larger CatBoost, XGBoost, KNN, or error-consistency workloads,
are the most likely to show a noticeable improvement. Small jobs may see little
improvement because device setup and data transfer still take time, and
CPU-only stages can remain a substantial part of the run. On a supported Mac,
GANDALF may use MPS in `auto` mode.

For example:

```shell
uv run python df-analyze.py \
    --df data/small_classifier_data.json \
    --target target \
    --mode classify \
    --classifiers xgb dummy \
    --device auto \
    --htune-trials 5 \
    --outdir ./device_results

uv run python df-embed.py \
    --data images.parquet \
    --modality vision \
    --device cuda \
    --out embeddings.parquet
```

If the installed copy of PyTorch cannot use an NVIDIA GPU, `df-analyze` and
`df-embed` can create a separate CUDA environment for the run:

```shell
uv run python df-analyze.py \
    --df data/small_classifier_data.json \
    --target target \
    --mode classify \
    --classifiers mlp dummy \
    --device cuda \
    --device-install auto \
    --htune-trials 5 \
    --outdir ./managed_cuda_results
```

The default, `--device-install never`, leaves the current Python environment
unchanged. Use `ask` to confirm setup interactively or `auto` to allow it
without a prompt. This requires a source checkout containing `pyproject.toml`
and `uv.lock`. The managed environment is stored in `.df-analyze-runtime` and
is reused until either of those files changes. It supplies CUDA-enabled
PyTorch; CatBoost and XGBoost use their own CUDA backends, so selecting only
those models does not trigger this setup.

### Verifying CUDA and GPU Visibility

Run these checks in the same terminal and environment that will run
`df-analyze`:

| Check | Command | Expected result |
|---|---|---|
| NVIDIA driver visibility | `nvidia-smi` | GPU name, driver version, memory information, and runtime status |
| PyTorch CUDA visibility | `python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"` | `True` when the installed PyTorch build can use CUDA; the second value is that build's CUDA runtime |
| CatBoost GPU visibility | `python -c "from catboost.utils import get_gpu_device_count; print(get_gpu_device_count())"` | A positive number when CatBoost sees one or more GPUs |

If `nvidia-smi` works but the PyTorch check prints `False`, the installed
PyTorch build may not support the available driver or CUDA runtime. For
installation and compatibility details, use the
[NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/),
the [PyTorch CUDA availability reference](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html),
and the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).


## Quick Start and Examples

Run a classification analysis on the data in `small_classifier_data.json`:

### Using Builtin Data

```bash
python df-analyze.py \
    --df=data/small_classifier_data.json \
    --outdir=./demo_results \
    --mode=classify \
    --target target \
    --classifiers knn lgbm rf lr sgd mlp dummy \
    --embed-select none linear lgbm \
    --feat-select wrap filter embed
```

should work and run quite quickly on the tiny toy dataset included in the repo.
This will produce a lot of terminal output.

Registered classifier tokens are:

```text
catboost xgb tabpfn dtree et knn lgbm rf lr sgd mlp kan svm gandalf dummy
```

Registered regressor tokens are:

```text
catboost xgb tabpfn dtree et knn lgbm rf elastic sgd mlp kan svm gandalf dummy
```

The `svm` token is retained for configuration compatibility, but the CLI
currently disables SVM training because of its runtime cost. The other tokens
can be evaluated when their dependencies and, where needed, model weights are
available.

TabPFN defaults to the v3 checkpoint. Use `--tabpfn-version v2.6` or
`--tabpfn-version v2.5` to select an older supported checkpoint. Before the
first download, accept the license for the selected checkpoint. The recommended
setup is the Prior Labs browser flow or a `TABPFN_TOKEN` from the Prior Labs
account page. Some `tabpfn` versions may instead report that a checkpoint is in
a gated Hugging Face repository. In that case, accept the terms for the named
repository and authenticate with `hf auth login` or a read-only `HF_TOKEN`.
Do not commit either token. For an offline machine, download the weights
separately and point `TABPFN_MODEL_CACHE_DIR` at that directory. See
[Prior Labs' model-access instructions](https://docs.priorlabs.ai/how-to-access-gated-models).
`df-analyze` checks that the selected cache is writable before a download.

Checkpoint licenses can differ and may change. Review the current license for
the selected version before commercial or production use; the
[TabPFN-3 model card and license](https://huggingface.co/Prior-Labs/tabpfn_3)
is the relevant page for the default v3 checkpoint.

TabPFN receives a separate raw-valued table with its categorical columns
identified, instead of the one-hot encoded matrix used by most other models.
Prior Labs documents the TabPFN-3 row/feature trade-off as 1 000 000 × 200,
100 000 × 2 000, or 1 000 × 20 000. `df-analyze` accepts an input only when it
fits at least one of those documented row/feature regimes; it does not infer
support for other shapes merely because their row-column product is small. The
[TabPFN-3 model card](https://huggingface.co/Prior-Labs/tabpfn_3) separately
states that predictive performance is not guaranteed above 2 000 features.
Inputs with more than 2 000 features are consequently accepted only inside the
documented 1 000 × 20 000 regime and produce an explicit experimental-regime
warning.
Passing these shape checks is not a memory or accuracy guarantee; compare
wide-input results against non-TabPFN baselines. See the current
[Prior Labs model limits](https://docs.priorlabs.ai/models) before interpreting
or publishing results. By default, `df-analyze` refuses to run TabPFN on the CPU
with more than 1 000 rows. Use CUDA or set
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` to allow a larger CPU run explicitly.

### Additional Model Backends

The following tokens add model backends beyond the original defaults:

| Token | Model |
|---|---|
| `catboost` | CatBoost gradient-boosted trees |
| `xgb` | XGBoost gradient-boosted trees |
| `tabpfn` | The selected TabPFN checkpoint |
| `dtree` | A scikit-learn decision tree |
| `et` | Scikit-learn extremely randomized trees |
| `kan` | A Kolmogorov-Arnold Network using PyKAN |

Pass these names to `--classifiers` or `--regressors` just like the original
model names:

```shell
python df-analyze.py \
    --df data/small_classifier_data.json \
    --target target \
    --mode classify \
    --classifiers catboost xgb dtree et kan dummy \
    --feat-select filter \
    --htune-trials 5 \
    --outdir ./additional_model_results
```

They use the same feature-set comparison, tuning, and final validation as the
other models. CatBoost, XGBoost, KAN, and TabPFN can use CUDA but do not require
it. TabPFN has the license, token, and model-download requirements described
above. No model is expected to be best for every dataset.

## Multi-Target Analysis

Use `--target outcome` for the usual single-target analysis. For several
targets, pass a comma-separated list to `--targets`:

```shell
# Several categorical outcomes
python df-analyze.py --df data.csv --targets outcome_a,outcome_b --mode classify

# Several numeric outcomes
python df-analyze.py --df data.csv --targets score_a,score_b --mode regress
```

The usual model and output defaults still apply. Keep these rules in mind:

- All targets in one run must have the same task type. Do not mix categorical
  and continuous targets.
- Quote the whole comma-separated value if a target name contains spaces.
- `--targets` takes precedence when both `--target` and `--targets` are given.
- Rows missing any target are removed.

`df-analyze` checks that every classification level has enough samples for the
requested validation folds. Every regression target must vary in the training,
holdout, and cross-validation partitions. If a valid split cannot be made, the
error names the target that caused the problem.

Feature selection is run for each target, then the results are combined with
Borda ranks or selection frequency:

```shell
--mt-agg-strategy borda
--mt-agg-strategy freq
--mt-top-k 25
```

Without `--mt-top-k`, every feature selected for at least one target is kept.
Models with native multi-output support fit the targets together; other models
fit one estimator per target. The default tuning search chooses one
configuration from the average per-target score. For regression, those tuning
scores are normalized so that a target with larger numeric values does not
dominate. Reported errors remain in the target's original units.

Final cross-validation uses up to five folds. A grouped run may use fewer folds
when fewer than five holdout groups are available, but a group is never split
between folds. The output column `final_cv_folds` records the number used.
Adaptive error analysis, when enabled, runs separately for each classification
target.

The main multi-target outputs are:

| Path | Description |
|---|---|
| `prepared/y.parquet` | Prepared target table; classification targets use encoded integers and regression targets retain their original units |
| `prepared/labels.parquet` | Per-target classification label maps used to translate encoded integers back to original labels |
| `features/associations/<target>/` | Per-target univariate association outputs |
| `features/predictions/<target>/` | Per-target univariate prediction outputs when prediction output is enabled |
| `results/results_report.md` | Overall final-evaluation report |
| `results/results_report_target_<target>.md` | Readable report for one target |
| `results/final_performances.csv` | Aggregate final-performance table |
| `results/final_performances_per_target.csv` | Compact per-target performance table |
| `results/performance_long_table_per_target.csv` | Long-form per-target metric table |
| `results/main_metric_by_target_acc.csv` or `results/main_metric_by_target_mae.csv` | Main metric for each classification or regression target |

## Using a `df-analyze`-formatted Spreadsheet

Run a classification analysis on the data in the file `spreadsheet.xlsx` with
configuration options and columns specifically formatted for `df-analyze`:

```shell
python df-analyze.py --spreadsheet spreadsheet.xlsx
```

Example of an Excel spreadsheet formatted for `df-analyze`:

![](./figures/xlsx1.png)

Another valid Excel spreadsheet:

![](./figures/xlsx2.png)

Example of a `.csv` spreadsheet formatted for `df-analyze`:

```csv
--outdir ./results
--target y
--mode classify
--categoricals s,x0
--classifiers knn lgbm dummy
--nan median
--norm robust
--feat-select wrap embed


s,x0,x1,x2,x3,y
male,0,0.739547,0.312496,1.129941,0
female,0,0.094421,0.817089,1.246469,1
unspecified,1,0.323189,0.008068,0.472934,0
male,2,0.570184,0.289003,1.176338,1
...
```

If you have not been introduced to command-line interfaces (CLIs) before,
this convention might seem a bit odd, but `df-analyze` primarily functions as
a CLI program. The logic is that CLI options (e.g. `--mode`) and their
parameters or values (e.g. the `classify` in `--mode classify`) are specified
one-per-line in the file top section / header, with spaces separating
parameters (e.g. the `knn lgbm dummy` parameters passed to the `--classifiers`
option), and with at least one empty line separating these options and
parameters from the actual tabular data.

Thus, the following is an **INVALIDLY FORMATTED** spreadsheet:

```csv
--outdir ./results
--target y
--mode classify
--categoricals s,x0
--classifiers knn dummy
--nan median
--norm minmax
--feat-select wrap filter none
s,x0,x1,x2,x3,y
male,0,0.739547,0.312496,1.129941,0
female,0,0.094421,0.817089,1.246469,1
unspecified,1,0.323189,0.008068,0.472934,0
male,2,0.570184,0.289003,1.176338,1
...
```

because no newlines (empty lines) separate the options from the data.



### Overriding Spreadsheet Options

When spreadsheet and CLI options conflict, then `df-analyze` will prefer the
CLI args. This allows a base spreadsheet to be set up, and for minor analysis
variants to be performed without requiring copies of the formatted data file.
So for example:

```shell
python df-analyze.py --spreadsheet sheet.xlsx --outdir ./results --test-val-size 0.2
python df-analyze.py --spreadsheet sheet.xlsx --outdir ./results --test-val-size 0.3
python df-analyze.py --spreadsheet sheet.xlsx --outdir ./results --test-val-size 0.4
```

would run three analyses with the options in `spreadsheet.xlsx` (or default
values) but with the holdout fraction differing for each run, regardless of
what is set for `--test-val-size` in `spreadsheet.xlsx`. Note that the same
output directory can be specified each time, as `df-analyze` will ensure that
all results are saved to a separate subfolder (with a unique hash reflecting
the unique combinations of options passed to `df-analyze`). This ensures data
should be overwritten only if the exact same arguments are passed twice (e.g.
perhaps if manually cleaning your data and re-running).

The parser still accepts the legacy `--nan` and `--norm` choices, including in
spreadsheet headers, but the current preparation paths use training-fitted
median imputation and robust normalization. Do not use those two options to
request a different preprocessing method in the current version.


## Embedding Functionality

`df-analyze` now supports the pre-processing of **image** and **text**
classification or regression datasets through the installed `df-embed`
command or the source-checkout `df-embed.py` Python script.

### Quickstart

The CLI help can be accessed locally by running

```bash
df-embed --help
```

From a source checkout, the equivalent command is:

```bash
python df-embed.py --help
```

Note that before any embedding is possible, you will need to download the
underlying [embedding models](#about-the-embedding-models) **once**. This can be done
with either of the commands:

```bash
python df-embed.py --download --modality nlp
python df-embed.py --download --modality vision
```

Embedding then uses `--device auto` by default, or an explicit CPU/CUDA
request:

```bash
python df-embed.py \
    --data my_data.parquet \
    --modality nlp \
    --device auto \
    --out my_data_embedded.parquet
```

**NOTE**: These models require enough memory for the input and model state.
CUDA is used when the installed PyTorch build and GPU support it; larger
embedding batches are more likely to benefit, while small batches may show
limited improvement. CPU execution is also supported. In either mode, the
input data and model state must fit in available memory.

On a Linux cluster, a CPU node with enough memory can also be used. Follow the
instructions to [build the container](#building-the-singularity-container), use
the included `run_python_with_home.sh` script, and pass absolute paths obtained
with `readlink` or `realpath`.


### About the Embedding Models

Internally, `df-analyze` uses two open-source HuggingFace zero-shot
classification models:
[SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) for
image data, and the large variant of the multilingual
[E5](https://huggingface.co/intfloat/multilingual-e5-large) text embedding
models. More specifically, the models are
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
and
[`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384),
which produce embedding vectors of size 1024 for each input text, and 1152
for each input image, respectively.

SigLip is a significant improvement on
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip), especially
for zero-shot classification (the main task in `df-analyze`). E5 uses an
[XLM-RoBERTa
backbone](https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta),
but is trained with a focus on producing quality zero-shot embeddings.



### Supported Dataset Formats

The ordinary dense `df-analyze` pipeline is intended for small to medium-sized
datasets (generally, under 200 features and under 200 000 or so samples).
[Large-scale feature downsampling](#large-scale-feature-downsampling) provides
separate paths for wider numeric or SVMlight matrices. The project strongly
aims to keep compute times under 24 hours (on a typical node on the [Niagara
cluster](https://docs.alliancecan.ca/wiki/Niagara)) for key operations
(embedding, predictive analysis). This means **any dataset to be embedded
should also generally be under about 200 000 samples**.

For embedding, `df-embed.py` supports CPU and CUDA inference through
`--device`, and, to not complicate data loading, currently requires a dataset
to fit in memory, loaded from a single, correctly-formatted `.parquet` file.

#### Image Data

For image classification data (`python df-embed.py --modality vision`), the
file must be a two-column table with the columns named "image" and "label".
The order of the columns is not important, but the "label" column must
contain integers in {0, 1, ..., c - 1}, where `c` is the number of class
labels for your data. The data type is not really important, however, if
the table is loaded into a Pandas DataFrame `df`, then running
`df["label"].astype(np.int64)` (assuming you have imported NumPy as `np`,
as is convention) should not alter the meaning of the data.

For image regression data (very rare), the file must be a two-column table
with the columns named "image" and "target". The order of the columns is
not important, but the "target" column must contain floating point values.
The floating point data type is not really important, however, if the table
is loaded into a Pandas DataFrame `df`, then running
`df["target"].astype(float)` should not raise any exceptions.

The "image" column must be of `bytes` dtype, and must be readable by PIL
`Image.open`. Internally, all we do, again assuming that the data is loaded
into a Pandas DataFrame `df`, is run:

```python
from io import BytesIO
from PIL import Image

df["image"].apply(lambda raw: Image.open(BytesIO(raw)).convert("RGB"))
```

to convert images to the necessary format. This means that if you load your
images using PIL `Image.open`, and you have a list of image paths (and a way
to infer the target from that path, e.g. `get_target(path: Path)`, then you
can convert your images to bytes through the use of `io` `BytesIO` objects,
and build your parquet file with just a few lines of Python:

```python
img: Image  # PIL Image
converted = []
targets = []

for path in my_image_paths:
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byts = buf.getvalue()
    converted.append(byts)
    targets.append(get_target(path))

df = DataFrame({"image": converted, "target": targets})
df.to_parquet("images.parquet")
```

#### Text Data

For text classification data (`python df-embed.py --modality nlp`), the
file must be a two-column table with the columns named "text" and "label".
The order of the columns is not important, but the "label" column must
contain integers in {0, 1, ..., c - 1}, where `c` is the number of class
labels for your data. The data type is not really important, however, if
the table is loaded into a Pandas DataFrame `df`, then running
`df["label"].astype(np.int64)` (assuming you have imported NumPy as `np`,
as is convention) should not alter the meaning of the data.

For text regression data (e.g. sentiment analysis, rating prediction), the
file must be a two-column table with the columns named "text" and
"target". The order of the columns is not important, but the "target"
column must contain floating point values. The floating point data type is
not really important, however, if the table is loaded into a Pandas
DataFrame `df`, then running `df["target"].astype(float)` should not raise
any exceptions.

The "text" column will have "object" ("O") dtype. Assuming you have loaded
your text data into a Pandas DataFrame `df`, then you can check that the
data has the correct type by running:

```python
assert df.text.apply(lambda s: isinstance(s, str)).all()
```

which will raise an AssertionError if a row has an incorrect type.

In order to keep compute times reasonable, it is best for text samples
to be at most a paragraph or two. I.e. the underlying model is not really
intended for efficient or effective document embedding. However, this
ultimately depends on the text language and it is hard to make general
recommendations here.

## Large-Scale Feature Downsampling

Feature downsampling reduces a very wide matrix before the usual univariate
analyses, feature selection, and model tuning. It is disabled by default. Use
`--feat-downsample` to choose a method and `--n-feat-downsample` to give either
a maximum feature count or retained fraction:

```shell
python df-analyze.py \
    --df wide.parquet \
    --target outcome \
    --mode classify \
    --feat-downsample auto \
    --n-feat-downsample 1000 \
    --outdir ./wide_results
```

For example, `--n-feat-downsample 500` keeps at most 500 features, while
`--n-feat-downsample 0.1` keeps 10 percent. The default maximum is 1000. The
available methods are:

```text
none auto random variance normalized-variance f-test mutual-info linear lgbm svd sparse-rp
rank-ensemble selector-ensemble stable-rank
```

`random` and `variance` do not use the target. `f-test`, `mutual-info`,
`linear`, and `lgbm` are supervised. The ensemble methods combine several
rankings, while `svd` and `sparse-rp` create new component features instead of
retaining named source columns.

The recommended method is `auto`. It leaves the data unchanged when the
requested number of features is already available, normally uses an F-test,
and uses repeated-subsample supervised `stable-rank` for extremely wide data.
If the target cannot support supervised screening, it falls back to
range-normalized variance.

Supervised methods use only training rows when scoring features. A separate
part of the training data is reserved for model tuning, and holdout or external
test rows are never used for feature scoring. Grouped data keep complete groups
together. If a safe supervised split cannot be made, `auto` falls back to
range-normalized variance; an explicitly requested supervised method stops with
an error.

For numeric tables that are too wide for the ordinary preparation path, use
`--large-feature-mode`:

```shell
python df-analyze.py \
    --df wide.parquet \
    --target outcome \
    --mode classify \
    --large-feature-mode \
    --feat-downsample rank-ensemble \
    --n-feat-downsample 1000 \
    --outdir ./large_feature_results
```

This mode splits rows and scores columns before building the selected training
and holdout matrices. Predictors must be numeric and finite. The source table
is still loaded as a pandas DataFrame and must fit in memory.

SVMlight input is available when the source matrix cannot be safely
materialized as a dense table:

```shell
python df-analyze.py \
    --df wide.svmlight \
    --target outcome \
    --mode classify \
    --feat-downsample f-test \
    --n-feat-downsample 500 \
    --outdir ./svmlight_results
```

Files ending in `.svm`, `.svmlight`, `.libsvm`, or `.binary` are recognized,
including gzip, bzip2, and xz compressed forms. The matrix remains sparse while
features are scored. SVMlight input requires feature downsampling and exactly
one target. Use `--svmlight-metadata` to append row-aligned clinical CSV, TSV,
JSON, or Parquet sidecars after imaging downsampling, and
`--svmlight-feature-map` to restore anatomical feature names. Feature maps may
mark prior-required imaging columns as protected. SVMlight row comments can be
checked against a BIDS/clinical ID column with
`--svmlight-sample-id-column`.

Auto mode also performs a bounded content check for SVMlight files with
nonstandard names, including extensionless text and gzip, bzip2, or xz files
such as `log1p.E2006.train.bz2`. For producers with a known feature-index
convention, pass `--svmlight-index-base zero` or `one`; absence of feature index
0 is inherently ambiguous. For example, the one-based E2006 regression
benchmark can be run with external validation using:

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

Results are written below `features/downsampling`. The main files are
`downsampling_report.md`, `downsampling.json`, `selected_features.csv`, and,
when scores are available, `feature_scores.csv`.

Sparse reports separate input scanning, train/test CSR loading, feature scoring,
and selected-feature materialization/scaling time. The score-chunk budget is not
a whole-process memory cap; source sparse matrices, temporary conversions,
full-length score vectors, and selected dense outputs remain additional.

Downsampling is a screening step and can discard useful interactions. When the
full analysis is practical, compare its holdout results with a run that does not
use downsampling. See
[Large-scale feature downsampling](docs/feature_downsampling.md) for the full
method descriptions and input restrictions.

## Usage on Compute Canada / Digital Research Alliance of Canada / Slurm HPC Clusters

It is *EXTREMELY* important that you only clone `df-analyze` into `$SCRATCH`, and
do all processing there. You have a very limited amount of space and absolute
number of files in your `$HOME` directory, and your login node will become
nearly unusable if you clone `df-analyze` there, or build the container in
`$HOME`. So just immediately `cd` to `SCRATCH` before doing any of the below.

### Building the Singularity Container

This should be built on a cluster that enables the `--fakeroot` option or on a
Linux machine where you have `sudo` privileges, and the same architecture as
the cluster (likely, x86_64).

First, clone the repository to `$SCRATCH`:

```bash
cd $SCRATCH
git clone https://github.com/stfxecutables/df-analyze.git
cd df-analyze
```

```bash
cd $SCRATCH/df-analyze/containers
./build_container_cc.sh
```

This will spam a lot of text to the terminal, but what you want to see at
the end is a message very similar to:

```txt
==================================================================
Container built successfully. Built container located at:
/scratch/df-analyze/df_analyze.sif
==================================================================
```

If you don't see this, or if somehow you see this message but there is no
`df_analyze.sif` in the project root, then the complete container build log
will be located in `df-analyze/containers/build.txt`. This `build.txt` file
should be included with any bug reports or if encountering any issues when
building the container.

You can perform a final additional sanity test of the container build by then
running the commands:

```bash
cd $SCRATCH/df-analyze/containers
./check_install.sh
```


You should see some output like:

```txt
Running script from: /scratch/[...]/df-analyze
Using Python 3.12.5
df-analyze 3.3.0
```

but with of course the final version number depending on which release you have
installed. Otherwise, there will be an error message and other information.

### Using the Singularity Container

If the singularity container `df_analyze.sif` is available in the project
root, then it can be used to run arbitrary python scripts with the [helper
script](run_python_with_home.sh)
included in the repo. E.g.

```bash
cd $SCRATCH/df-analyze
./run_python_with_home.sh "$(realpath my_script.py)"
```

**HOWEVER** this will frequently cause errors about files not being found.
This has to do with aliasing and the complex file systems on Compute Canada
and how these interact with path-mounting in Apptainer, but the solution is
to **ALWAYS WRAP PATHS WITH THE `realpath` COMMAND**. E.g.

```bash
./run_python_with_home.sh df-embed.py \
    --modality vision \
    --data "$(realpath my_images.parquet)" \
    --out "$(realpath embedded.parquet)"
```

this should be done if running a command in a login-node, or if making a job
script to submit to the SLURM scheduler.

# Analysis Pipeline

The main data preparation and analysis steps are:

1. Load the data and create the raw training and holdout partitions.
1. Infer feature types from the training rows.
1. Fit preprocessing on the training rows and apply it to the holdout rows.
1. Optionally downsample a very wide feature matrix.
1. Run the univariate feature analyses.
1. Optionally select features.
1. Tune the requested models.
1. Evaluate the tuned models on the final holdout.
1. Optionally run adaptive error or error consistency analyses.

In pseudocode (which closely approximates the code in the `main()` function of
[`df-analyze.py`](df-analyze.py)):

```python
    options = get_options()
    df = options.load_df()

    train_rows, test_rows = raw_train_test_indices(df, options)
    inspection = inspect_data(df.iloc[train_rows], options)
    prepared = prepare_data(df, inspection, train_rows, test_rows)

    for train, test in prepared.get_splits():
        associations = target_associations(train)
        predictions = univariate_predictions(train)
        selected = select_features(train, associations, predictions, options)
        tuned = tune_models(train, selected, options)
        results = eval_tuned(test, tuned, selected, options)
```

### Feature Type and Cardinality Inference

Features are checked, in order of priority, for features that cannot be used
by `df-analyze`. Unusable features are features which are:

1. Constant (all values identical or identical except NaNs)
2. Sequential (autocorrelated) datetime data
3. Identifiers (all values unique and not continuous / floats)

Then, features are identified as one of:

1. Binary
2. Ordinal
3. Continuous
4. Categorical

based on a number of heuristics relating to the unique values and counts of
these values, and the string representations of the features. These
heuristics are made explicit in code
[here](src/df_analyze/preprocessing/inspection/inference.py).

### Data Preparation

Input data is transformed so that it can be accepted by most generic ML
algorithms and/or Python data science libraries (but particularly,
[NumPy](https://numpy.org/),
[Pandas](https://pandas.pydata.org/docs/user_guide/10min.html#min),
[scikit-learn](https://scikit-learn.org/stable/index.html),
[PyTorch](https://pytorch.org/), and
[LightGBM](https://lightgbm.readthedocs.io/en/stable/)). This means the
**_raw_** input data $\mathcal{D}$ (specified by `--df` or `--spreadsheet`
argument) is represented as

$$\mathcal{D} = (\mathbf{X}, y) = \texttt{(X, y)},$$

 where
`X` is a Pandas `DataFrame`. For a single-target analysis, `y` is a Pandas
`Series`; for a multi-target analysis, it is a Pandas `DataFrame`.

1. Data Loading
   1. Type Conversions
   1. NaN unification (detecting less common NaN representations)
1. Data Cleaning
   1. Remove samples with NaN in target variable
   1. Remove junk features (constant, timeseries, identifiers)
   1. NaNs: remove or add indicators and interpolate
   1. [Categorical deflation](#categorical-deflation) (replace undersampled
      classes / levels with NaN)
1. Feature Encoding
   1. Binary categorical encoding
      1. represented as single [0, 1] feature if no NaNs
      1. single NaN indicator feature added if feature is binary plus NaNs
   1. One-hot encoding of categoricals (NaN = one additional class / level)
   1. Ordinals treated as continuous
   1. Robust normalization of continuous features
2. Target Encoding
   1. Categorical [targets are deflated](#categorical-target-deflation) and
      label encoded to values in $[0, n]$
   2. A single continuous target keeps the original public behavior and is
      robustly normalized using its 2.5th and 97.5th percentiles
   3. Multiple continuous targets are converted to numeric values and kept in
      their original units

#### Categorical Deflation

Categorical variables will frequently contain a large number of classes that
have only a very small number of samples.

For example, a small, geographically representative survey of households
(e.g. approximately 5000 samples) might contain region / municipality
information. Regions or municipalities corresponding to large cities might
each have over 100 samples, but small rural regions will likely be sampled
less than 10 or so times each, i.e., they are *undersampled*. Attempting to
generalize from any patterns observed in these undersampled classes is
generally unwise (undersampled levels in a categorical variable are sort
of the categorical equivalent of statistical noise).

In addition, leaving these undersampled levels in the data will usually
significantly increase compute costs (each class of a categorical, in most
encoding schemes, will increase the number of features by one), but encourage
overfitting or learning of spurious (ungeneralizable) patterns. It is thus
wise, usually, for both computational and generalization reasons, to exclude
these classes from the categorical variable (e.g. replace with NaN, or a
single "other" class).

In `df-analyze`, we **automatically deflate categorical variables based on a
threshold of 20 samples**, i.e. classes with less than 20 samples are
remapped to the "NaN" class. This is probably not aggressive enough for most
datasets, and, for some features and smaller datasets, perhaps overly
aggressive. However, if later feature selection is used, this selection is
done on the one-hot encoded data, and so useless classes will be excluded in
a more principled way there. The choice of 20 is thus (hopefully) somewhat
conservative in the sense of not prematurely eliminating information, most of
the time.

##### Categorical Target Deflation

As above, target categorical variables are deflated, except when a target
class has less than 30 samples. This deflation arguably should be *much* more
aggressive: when doing e.g. 5-fold analyses on a dataset with such a target
variable, each test fold would be expected to be 20% of the samples, so about
6 representatives of this class. This is highly unlikely to result in
reliable performance estimates for this class, and so only introduces noise
to final performance metrics.

For a single target, classes with 20 or fewer samples are removed. This matches
the original cleaning behavior and avoids folds with too few examples.

Multi-target classification is handled differently. Removing a row because one
target has a rare class would also remove valid labels from the other targets,
so low-support classes are retained and reported instead. The run continues
only when valid training and validation splits can still be constructed.

### Data Splitting

The data $\mathcal{D} = (\mathbf{X}, y)$ is immediately split into
non-overlapping sets
$\mathcal{D} = (\mathcal{D}\_\text{train}, \mathcal{D}\_\text{test})$, where
$\mathcal{D}\_\text{train} = (\mathbf{X}\_{\text{train}}, y\_{\text{train}})$ and
$\mathcal{D}\_\text{test} = (\mathbf{X}\_{\text{test}}, y\_{\text{test}})$. By
default $\mathcal{D}\_\text{test}$ is chosen to be a (stratified) random 40%
of the samples. All selection and tuning is done only on
$\mathcal{D}\_\text{train}$, to prevent circular analysis / double-dipping /
leakage.


### Univariate Feature Analyses

1. Univariate associations
2. Univariate predictive performances
   1. classification task / categorical target
      - tune
        [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
        (an approximation of linear SVM and/or Logistic Regression)
      - report 5-fold mean accuracy, AUROC, sensitivity and specificity
   2. regression task / continuous target
      - tune
        [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)
        (an approximation of regularized linear regression)
      - report 5-fold mean MAE, MSqE, $R^2$, percent explained variance, and
        median absolute error

### Feature Selection

- Use filter methods
   - Remove features with minimal univariate relation to target
   - Keep features with largest filter metrics
- Wrapper (stepwise) selection
- Filter selection

#### Redundancy-Aware Feature Selection \*\*[NEW]\*\*

<!-- $\boldsymbol{X}$
\boldsymbol{X}
$\symbfit{X}$
\symbfit{X} -->


Given training data
$\mathcal{D}\_\text{train} = (\mathbf{X}\_{\text{train}}, y\_{\text{train}})$
and a univariate estimator $f$ ("*selector*") with
suitable fixed default hyperparameters, redundancy-aware feature selection
performs forward stepwise selection, removing from consideration all features
with similar performances at each step. This similarity is controlled by a
*equivalence threshold* $\tau$ (default 0.005, i.e. an invisible difference
after rounding to two decimal places).

At each step of stepwise selection, the best candidate feature
$\symbfit{x}^{\star}$ produces some loss $\mathcal{L}^{\star}$. This defines a
*redundant set* of features,
$R = \\{ \symbfit{x} \text{ s.t. } | \mathcal{L}\big( f(\symbfit{x}), y \big) - \mathcal{L}^{\star} | < \tau \\}$.
That is, all features $\symbfit{x}$ in the redundant set are such that adding
$\symbfit{x}$ instead of $\symbfit{x}^{\star}$ to the previous iteration
feature pool produces performance that is considered equivalent at threshold
$\tau$. At the next iteration of redundant stepwise selection, rather than
just eliminating $\symbfit{x}^{\star}$ from consideration, instead all
features in $\symbfit{R}$ are also greedily eliminated.


**Algorithm**

> #### **Initialization $(i = 0)$**
>
> - define $\symbfit{F}_i = \\{\symbfit{x}_1, \dots, \symbfit{x}_p\\}$ to be the *candidate feature pool* at iteration $i$
> - define $\symbfit{X}_i^{\star} = \emptyset$, to be the *selected* features at iteration $i$
> - define $\symbfit{X}_i^R = \emptyset$ to be the *redundant* features at iteration $i$
> - For the *selector*, choose supervised estimator (classifier, regressor) $f$
>   and suitable default constant hyperparameters for $f$, such that $f$ is fit
>   to predict $y$ from any feature subset $\symbfit{X}$, i.e. we aim to fit $f$
>   such that a loss $\mathcal{L}\big( f(\symbfit{X}), y \big)$ is minimized.
>
> #### **Iteration $(i > 0)$**
>
> 1. For each feature $\symbfit{x} \in \symbfit{F}\_i$, define
>    $\symbfit{X}_{i} = \symbfit{X}_i^{\star} \cup \\{ \symbfit{x} \\}$ to
>    be the *candidate feature set*.
> 2. Define $\mathcal{L}\_{i} = \\{ \mathcal{L} \big( f( \symbfit{X}_i ), y \big) | \symbfit{x} \in \symbfit{F}_i \\}$ to be the set of candidate losses / performances
>    of each feature set
> 3. Define $\mathcal{L}\_i^{\star} = \min \mathcal{L}_i$. The feature $\symbfit{x}^{\star}$
>    producing $\mathcal{L}_i^{\star}$ is the best new candidate feature.
> 4. Set $\symbfit{R} = \big\\{ \symbfit{x} : | \mathcal{L}_k - \mathcal{L}_i^{\star} | \le \tau \text{ and } \symbfit{x} \in \symbfit{F}_i \big\\}$ to be the set of features redundant to $\symbfit{x}^{\star}$
> 5. Set $\symbfit{F}_{i+1} = \symbfit{F}_i - \symbfit{R}$ (remove redundant features from candidate pool)
> 6. Set $\symbfit{X}\_{i+1}^{\star} = \symbfit{X}_{i}^{\star} \cup \\{ \symbfit{x}^{\star} \\}$ (add selected feature to "selected" pool)
> 7. Set $\symbfit{X}\_{i+1}^R = \symbfit{X}_i^R \cup \symbfit{R}$ (update redundant pool)
> 8. Continue iterating $i$ until $\symbfit{F}_i = \emptyset$ or maximum $i$ is reached. The final selected features are defined by $\symbfit{X}_i^{\star}$.



### Hyperparameter Tuning

- Bayesian (Optuna) hyperparameter optimization with internal 5-fold
  validation

### Final Validation

- Final k-fold of model tuned and trained on selected features from $\mathcal{D}_\text{train}$
- Final evaluation of trained model on $\mathcal{D}_\text{test}$

### Adaptive Error Analysis

Pass `--adaptive-error` to estimate the chance that each classification
prediction is wrong. Unlike accuracy, which summarizes the whole holdout set,
adaptive error gives each row its own estimated error rate:

```shell
python df-analyze.py \
    --df data/small_classifier_data.json \
    --target target \
    --mode classify \
    --classifiers lgbm xgb dummy \
    --feat-select filter \
    --htune-trials 5 \
    --adaptive-error \
    --aer-oof-folds 5 \
    --aer-bins 20 \
    --outdir ./adaptive_error_results
```

The confidence-to-error mapping is learned from out-of-fold predictions on the
training data. Holdout labels are used only to report how well that mapping
worked. In a multi-target classification run, each target is analyzed
separately.

#### How Adaptive Error Works

1. Refit the tuned model in several training folds and collect out-of-fold
   predictions.
2. Calibrate the class probabilities when a supported calibrator improves them.
3. Compare the confidence measures available for that model. In `auto` mode,
   choose the one with the best cross-fitted Brier score.
4. Learn the relationship between confidence and observed error on the
   out-of-fold predictions.
5. Apply that learned relationship to the holdout predictions.
6. Report coverage and error thresholds with one-sided Clopper-Pearson bounds.

#### Options

The main options are:

| Flag | Default | Description |
|---|---:|---|
| `--adaptive-error` | `False` | Enable AER analysis |
| `--aer-oof-folds` | `5` | OOF splits used for AER fitting and cross-fitting |
| `--aer-bins` | `20` | Nominal number of confidence bins |
| `--aer-min-bin-count` | `10` | Minimum observations before bins are merged or reduced |
| `--aer-prior-strength` | `2.0` | Strength of shrinkage toward the global error rate |
| `--no-aer-smooth` | off | Disable the default local smoothing |
| `--aer-monotonic` | `False` | Enforce a monotonic confidence-to-error mapping |
| `--aer-adaptive-binning` | `False` | Use quantile-like adaptive instead of fixed-width bins |
| `--aer-confidence-metric` | `auto` | Select a confidence signal; `auto` uses cross-fitted Brier score |
| `--aer-nmin` | `1` | Minimum accepted observations at a risk-controlled threshold |
| `--aer-target-error` | `0.05` | Target error rate for risk-control summaries |
| `--aer-alpha` | `0.05` | Significance level for the one-sided error bound |
| `--aer-top-k` | `0` | Analyze at most the top *k* usable base models; `0` means all |
| `--no-preds` | off | Replace large per-sample prediction outputs with placeholders |

Available base-model confidence signals are:

| Metric | Meaning |
|---|---|
| `proba_margin` | Margin between the leading class probabilities |
| `tree_vote_agreement` | Agreement among individual tree votes |
| `tree_leaf_support` | Training support represented by tree leaves |
| `knn_vote` | Nearest-neighbor vote agreement |
| `knn_dist_weighted` | Distance-weighted nearest-neighbor confidence |
| `knn_min_dist` | Confidence derived from nearest-neighbor distance |
| `auto` | Select the best available signal by cross-fitted Brier score |

Use `--aer-ensemble` to compare combinations of the eligible models. Select
specific combinations with `--aer-ensemble-strategies`:

```shell
--aer-ensemble \
--aer-ensemble-strategies min_aer topn calibration_aware
```

Adaptive error is available only for classification models that provide usable
class probabilities. Dummy models are skipped.

#### Outputs

Results are written below `results/adaptive_error`. The main files contain the
model ranking, the learned confidence/error relationship, one estimated error
rate per holdout row, reliability bins, and coverage/accuracy summaries. See
`python df-analyze.py --help` for every option.

For a multi-target run, each target has its own sanitized subdirectory below
`results/adaptive_error`; with multiple external test sets, `testXX` is added
before the target directory. The cross-model files at each AER base directory
include:

| Path | Purpose |
|---|---|
| `run_config.json` | AER settings, selected models, and run metadata |
| `tables/models_ranked.csv` | Ranking and folder location of analyzed base models |
| `tables/aer_metrics_by_model.csv` | Cross-model error-quality metrics |
| `plots/confidence_vs_expected_error_compare.png` | Confidence-to-error comparison across models |
| `predictions/test_per_sample_multi_model.csv` | Model-specific AER columns aligned to the same holdout rows |

Each `models/<model-slug>/` directory can contain:

| Path | Purpose |
|---|---|
| `metadata/proba_calibrator.json` | Selected probability-calibration method |
| `metadata/confidence_metric_selection.json` | Selected confidence signal and candidate Brier scores |
| `metadata/adaptive_error_metrics.json` | Global test error and AER calibration summary |
| `tables/oof_confidence_error_bins.csv` | OOF bins used to learn the mapping |
| `tables/test_confidence_error_bins.csv` | Holdout behavior of that mapping |
| `tables/test_error_reliability_bins.csv` | Calibration-style reliability table |
| `tables/coverage_accuracy_curve.csv` | Selective accuracy as progressively higher-risk rows are rejected |
| `tables/coverage_summary.csv` | Selected operating points from the full coverage curve |
| `tables/clinician_view.csv` | Row ID, labels, `aer_pct`, and target-error flag |
| `predictions/oof_per_sample.csv` | Row-level diagnostics for learning the mapping |
| `predictions/test_per_sample.csv` | Main row-level holdout output |
| `reports/clinician_view.md` | Short report for non-technical readers |

The main columns in `predictions/test_per_sample.csv` are:

| Column | Meaning |
|---|---|
| `row_id` | Original row index |
| `y_true`, `y_pred` | Encoded true and predicted class IDs |
| `y_true_label`, `y_pred_label` | Decoded labels when a label map is available |
| `correct` | `1` for a correct prediction, otherwise `0` |
| `confidence` | Selected confidence signal after its transformation |
| `aer`, `aer_pct` | Estimated error probability as a fraction and percentage |
| `flag_gt_target_error` | `1` when `aer` meets or exceeds `--aer-target-error` |
| `p_max`, `p_2nd`, `p_margin` | Diagnostics from calibrated class probabilities |
| `p_pred`, `p_pred_margin` | Probability diagnostics for the predicted class |

In `adaptive_error_metrics.json`, `global_error_test` is the ordinary holdout
error rate. `brier_error_test` and `ece_error_test` measure the quality of the
per-row error estimates; smaller values are better.

If the AER directory is absent, first confirm that the task is classification,
that `--adaptive-error` was passed, and that at least one non-dummy model
provided usable probabilities. Placeholder prediction files indicate that
`--no-preds` was used. Use `--aer-adaptive-binning` to switch to quantile-based
bins or increase `--aer-min-bin-count` to merge sparse bins more aggressively.

### Error Consistency

Error consistency (EC) asks whether models trained on slightly different rows
make similar mistakes on the same samples. df-analyze completes feature
selection and tuning first. It then repeats K-fold splitting on the training
data, fits one model per fold, and evaluates every fitted model on the same
external holdout.

This small classification example uses 2 folds and 2 repetitions, so each
configuration is refitted 4 times:

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

The regression example uses the small Forest Fires dataset included in this
repository:

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

Omit `--ec-methods` to calculate all seven regression methods. The short list
above is only meant to make a first run quicker.

Two profiles provide settings used by the EC reference experiments:

```shell
# 80/20 holdout, 5 folds, 10 repetitions, fixed model seeds
--ec-profile classification-paper

# 80/20 holdout, 5 folds, 50 repetitions, fixed model seeds
--ec-profile regression-paper
```

Explicit CLI or spreadsheet values override individual profile settings. These
profiles reproduce the split and repetition settings, not the original
datasets, preprocessing, models, or published result tables.

The normal default is 5 folds and 5 repetitions: 25 refits for every target,
model, and selected feature set. The classification profile uses 50 refits; the
regression profile uses 250. Start with 2 folds and 2 repetitions to check the
workflow and estimate runtime. `--ec-output-detail summary` reduces the files
kept in memory and written to disk, but it does not reduce the number of model
fits.

EC uses the holdout already created by df-analyze; it does not create another
one. Without a profile, `--test-val-size` defaults to `0.4`. Add
`--test-val-size 0.2` when an 80/20 split is required.

Classification EC is the intersection-over-union of two models' error sets.
Regression EC compares their residuals. Some regression methods are best at 1
and others at 0, so read the `optimal_value` and `optimization_direction`
columns in `summary.csv`.

The default `--ec-holdout-role test` calculates EC for reporting but leaves the
ranking and EC/performance correlation files empty. Use
`--ec-holdout-role validation` only when the shared holdout is separate
validation or audit data. Do not choose a model from final-test results.

df-analyze creates a dataset/run folder below `--outdir`. EC results are in
`results/error_consistency` inside that run folder. If adaptive error and error
consistency are both enabled, a joined
`results/adaptive_error/tables/risk_stability_report.csv` is also produced.
Start with `summary.csv`, `performance_summary.csv`, `trial_failures.csv`, and
`selection_guard.csv`. See the
[error-consistency guide](docs/error_consistency.md) for the formulas, method
names, runtime guidance, checkpoints, and interpretation. See
[command-line arguments](docs/arguments.md) for every option and
[program outputs](docs/program_outputs.md) for the file layout.


# Program Outputs

The output directory structure is as follows:

```
📂 ./my_output_directory
└── 📂 fe57fcf2445a2909e688bff847585546/
    ├── 📂 features/
    │   ├── 📂 associations/
    │   ├── 📂 descriptions/
    │   ├── 📂 downsampling/
    │   └── 📂 predictions/
    ├── 📂 inspection/
    ├── 📂 prepared/
    ├── 📂 results/
    │   ├── 📂 adaptive_error/       # when --adaptive-error is enabled
    │   └── 📂 error_consistency/    # when --error-consistency is enabled
    ├── 📂 selection/
    │   ├── 📂 embed/
    │   ├── 📂 filter/
    │   └── 📂 wrapper/
    └── 📂 tuning/
```

The directory `./my_output_directory` is the directory specified by the
`--outdir` argument.

There are 4 main types of files:

1. Markdown reports (`*.md`)
1. Plaintext / CSV table files (`*.csv`)
1. Compressed Parquet tables (`*.parquet`)
1. Python object representations / serializations (`.json`)

Markdown reports (`*_report.md`) should be considered the main outputs: they
include text describing certain analysis outputs, and inlined tables of key
numerical results for that portion of analysis. The inline tables in each
Markdown report are saved in the same directory of the report always as
plaintext CSV (`*.csv`) files, and also occasionally additionally as a
Parquet file (`*.parquet`). This is because CSV is inherently lossy and, to
be blunt, basically a [trash format for representing tabular
data](https://haveagreatdata.com/posts/why-you-dont-want-to-use-csv-files/).
However, it is human-readable and easy to import into common spreadsheet
tools (Excel, Google Sheets, LibreOffice Calc, etc.).

The `.json` files are largely for internal use and in general should not need
to be inspected by the end-user. However, `.json` was chosen over, e.g.,
`.pickle`, since `.json` is at least fairly human-readable, and in
particular, the [`options.json` file](#📂-hashed-subdirectories) allows for
expanding main output tables with additional columns reflecting the program
options across [multiple runs](#overriding-spreadsheet-options).

## Order of Reading

The subdirectories should generally be read / inspected in the following order:

```
📂 inspection/
📂 prepared/
📂 features/
📂 selection/
📂 tuning/
📂 results/
```

## Subdirectories

### 📂 Hashed Subdirectories

```
📂 fe57fcf2445a2909e688bff847585546/
├── 📂 features/
│   ...
├── features_renamings.md
├── options.json
└── terminal_outputs.txt
```

This directory is named after a unique hash of all the options used for a
particular invocation / execution of the `df-analyze` command.

The `feature_renamings.md` file indicates which features have been renamed due
to problematic characters or duplicate feature names.

The `options.json` file is a `.json` representation of the specific invocation
or spreadsheet options. `terminal_outputs.txt` captures the run's terminal
output. The options hash allows multiple sets of outputs from different
options to be placed automatically in the same `--outdir` top-level directory,
e.g. as mentioned [above](#overriding-spreadsheet-options).

So for example, running multiple options combinations to the same output
directory will make something like:

```
📂 ./my_output_directory
├── 📂 ecc2d425d285807275c0c6ae498a1799/
├── 📂 fe57fcf2445a2909e688bff847585546/
└── 📂 7c0797c3e6a6eebf784f33850ed96988/
```

### `📂 inspection`

```
📂 inspection/
├── inferred_types.csv
└── short_inspection_report.md
```

This contains the inferred cardinalities (e.g. continuous, ordinal, or categorical)
of each feature, as well as the decision rule used for each inference. Features
with ambiguous cardinalities are also coerced to some cardinality (usually ordinal,
since categorical variables are often low-information and increase compute costs),
and this is detailed here.

#### Destructive Data Changes

Nuisance features (timeseries features or non-categorical
datetime data, unique identifiers, constant features) are automatically
removed by `df-analyze`, and those destructive data changes are documented
here.

[Deflated categorical variables](#categorical-deflation) are documented here
as well.


### `📂 prepared`

```
📂 prepared/
├── info.json
├── idx_tests.json
├── labels.parquet
├── preparation_report.md
├── X.parquet
├── X_cat.parquet
├── X_cont.parquet
├── X_tabpfn.parquet
└── y.parquet
```

- `preparation_report.md`
  - shows compute times for processing steps, and documents changes to the data
    shape following encoding, deflation, and dropping of target NaN values
- `labels.parquet`
  - target label encodings for one or more classification targets
- `idx_tests.json`
  - row indices used for the saved train and test partitions
- `X.parquet`
  - the final encoded complete data (categoricals and numeric)
- `X_cat.parquet`
  - the original (unencoded) categoricals
- `X_cont.parquet`
  - the continuous features (normalized and NaN imputed)
- `X_tabpfn.parquet`
  - the separate raw-valued/categorical predictor view used by TabPFN
- `y.parquet`
  - the final encoded target variable
- `info.json`
  - serialization of internal `InspectionResults` object


### `📂 features`

```
📂 features/
├── 📂 associations/
├── 📂 descriptions/
├── 📂 downsampling/
└── 📂 predictions/
```

Data for univariate analyses of all features.

#### `📂 downsampling`

This directory is populated when `--feat-downsample` is enabled:

```
📂 downsampling/
├── downsampling.json
├── downsampling_report.md
├── feature_scores.csv
└── selected_features.csv
```

- `downsampling_report.md`
  - readable summary of the requested/resolved method, dimensions, screening
    design, timings, and selected features
- `downsampling.json`
  - machine-readable form of the same metadata
- `selected_features.csv`
  - selected source features or generated component names in output order
- `feature_scores.csv`
  - saved ranking scores when the selected method produces them

Fold suffixes are added when an analysis has multiple external-test splits.
`feature_scores.csv` is omitted for methods without feature scores.

#### `📂 associations`

```
📂 associations/
├── associations_report.md
├── categorical_features.csv
├── categorical_features.parquet
├── continuous_features.csv
└── continuous_features.parquet
```

- `associations_report.md`
  - tables of feature-target associations
- `categorical_features.csv`
  - plaintext table of categorical feature-target associations
- `categorical_features.parquet`
  - Parquet table of categorical feature-target associations
- `continuous_features.csv`
  - plaintext table of continuous feature-target associations
- `continuous_features.parquet`
  - Parquet table of continuous feature-target associations

#### `📂 descriptions`

```
📂 descriptions/
├── categorical_features.csv
├── continuous_features.csv
└── target.csv
```

- `categorical_features.csv`
  - table where each row describes a categorical feature
  - columns are, for each feature:
    ```
    "n_levels":  # number of levels / categories / classes
    "nans":      # NaN count
    "nan_freq":  # NaN frequency
    "med_freq":  # Median of level frequencies
    "min_freq":  # Frequency of least-frequent class
    "max_freq":  # Frequency of most-frequent class
    "min_name":  # Name of least common class
    "max_name":  # Name of most common class
    "heterog":   # chi-square / heterogeneity
    "heterog_p": # chi-square p-value
    "n_entropy": # normalized entropy (closer to max of 1 = more uniform)
    ```
- `continuous_features.csv`
  - table where each row describes a continuous feature
  - columns are, for each feature:
    ```
    "min":       # minimum value
    "mean":      # mean value
    "max":       # maximum value
    "sd":        # standard deviation
    "p05":       # 5th percentile value
    "median":    # median value
    "p95":       # 95th percentile value
    "iqr":       # interquartile range
    "skew":      # skewness / asymmetry
    "skew_p":    # p-value for test if skew is different from Gaussian
    "kurt":      # kurtosis / tailedness
    "kurt_p":    # p-value for test if kurtosis is different from Gaussian
    "entropy":   # differential entropy (scipy.stats, default args)
    "nan_freq":  # proportion of NaN values
    ```
- `target.csv`
  - same format as either of above, depending on if the task is classification
    or regression

#### `📂 predictions`

```
📂 predictions/
├── categorical_features.csv
├── categorical_features.parquet
├── continuous_features.csv
├── continuous_features.parquet
└── predictions_report.md
```

- `categorical_features.csv`
  - plaintext table of 5-fold predictive performances of each categorical feature
- `categorical_features.parquet`
  - Parquet table of 5-fold predictive performances of each categorical feature
- `continuous_features.csv`
  - plaintext table of 5-fold predictive performances of each continuous feature
- `continuous_features.parquet`
  - Parquet table of 5-fold predictive performances of each continuous feature
- `predictions_report.md`
  - summary tables of all feature predictive performances

**Note**: for "large" datasets
([currently](src/df_analyze/_constants.py),
greater than 1500 samples) these predictions are made using a small (1500
samples) subsample of the full data, for compute time reasons.

For continuous targets (i.e. regression), the subsample is made in a
representative manner by taking a stratified subsample, where stratification
is based on discretizing the continuous target variable into 5 bins (via
scikit-learn `KBinsDiscretizer` and `StratifiedShuffleSplit`, respectively).

For categorical targets (e.g. classification), the subsample is a "viable
subsample" (see `viable_subsample` in
[`prepare.py`](src/df_analyze/preprocessing/prepare.py))
that first ensures all target classes have the minimum number of samples
required to avoid deflation and/or problems with 5-fold splits eliminating
a target class.

### `📂 selection`

```
📂 selection/
├── 📂 embed/
├── 📂 filter/
└── 📂 wrapper/
```

Data describing the features selected by each feature selection method.


#### `📂 embed`

```
📂 embed/
├── <model>_embed_selection_data.json
└── <model>_embedded_selection_report.md
```

- `<model>_embedded_selection_report.md`
  - summary of features selected by (each) embedded model
- `<model>_embed_selection_data.json`
  - feature names and importance scores for `<model>`

#### `📂 filter`

```
📂 filter/
├── association_selection_report.md
└── prediction_selection_report.md
```

- `association_selection_report.md`
  - summary of features selected by univariate associations with the target
  - also includes which measure of association was used for selection
- `prediction_selection_report.md`
  - summary of features selected by univariate predictive performance
  - also includes which predictive performance metric was used for selection

**Note**: Feature importances are not included here, as these are already
available in the [`features` directory](#📂-features).

#### `📂 wrapper`

```
📂 wrapper/
├── wrapper_selection_data.json
└── wrapper_selection_report.md
```

- `wrapper_selection_data.json`
  - feature names and predictive performance of each upon selection
- `wrapper_selection_report.md`
  - summary of features selected by wrapper (stepwise) selection method

### `📂 tuning`

```
📂 tuning/
└── tuned_models.csv
```

- `tuned_models.csv`
  - table of all tuned models for each feature selection method, including
    final performance and final selected hyperparameters (as a .json field)


### `📂 results`

```
📂 results/
├── 📂 adaptive_error/
├── 📂 error_consistency/
├── eval_htune_results_jsonpickle.json
├── final_performances.csv
├── final_performances_per_target.csv
├── main_metric_by_target_<metric>.csv
├── performance_long_table.csv
├── performance_long_table_per_target.csv
├── prediction_results.json
├── results_report_target_<target>.md
├── results_report.md
├── run_timing.json
├── X_test.csv
├── X_train.csv
├── y_test.csv
└── y_train.csv
```

- `final_performances.csv` and `performance_long_table.csv` [TODO: make one
  of these wide table]
  - final summary table of all performances for all models and feature
    selection methods
- `final_performances_per_target.csv`,
  `performance_long_table_per_target.csv`,
  `main_metric_by_target_<metric>.csv`, and
  `results_report_target_<target>.md`
  - optional multi-target long-form, main-metric, and readable per-target
    results
- `adaptive_error` and `error_consistency`
  - optional analysis directories described in
    [Adaptive Error Analysis](#adaptive-error-analysis) and
    [Error Consistency](#error-consistency)
- `prediction_results.json`
  - dictionary of all actual predictions (predicted classes in
    classification, predicted target values in regression) and, if
    classification and available for the model, predicted probabilities
  - can be loaded externally with `json.loads(path.read_text())` using Python
    stdlib [`json`](https://docs.python.org/3/library/json.html#module-json),
    and where `path` is a
    [Pathlib](https://docs.python.org/3/library/pathlib.html#module-pathlib)
    `Path` pointing to `prediction_results.json`
    - the `preds_train`, `preds_test`, `probs_train`, and `probs_test` fields
    are Python lists that can be converted to NumPy with `np.array`
    - Dtype information for above conversions is stored in `preds_dtype` and
      `probs_dtype` fields
    - for probability arrays such as `probs_test`, after making a NumPy
      ndarray, the entry at [*i*, *j*] corresponds to the predicted probability
      for sample *i* and label *j*
    - the original meaning of the labels prior to encoding is stored in
      `labels.parquet`, in the [`prepared` folder](#📂-prepared)
- `results_report.md`
  - readable report (with wide-form tables of performances) of above
    information
- `X_test.csv`
  - predictors used for final holdout and k-fold evaluations
- `X_train.csv`
  - predictors used for training and tuning
- `y_test.csv`
  - target samples used for final holdout and k-fold evaluations
- `y_train.csv`
  - target samples used for training and tuning
- `eval_htune_results_jsonpickle.json`
  - serialization of final results object (not human readable, for internal
    use)
- `run_timing.json`
  - run status and total time, separate planned and actually resolved device
    decisions, per-model-task device records, runtime fold counts, model
    failures, partial analysis failures, and error-consistency backend
    information

## Complete Listing

One representative tree-structure of the baseline outputs is as follows.
Additional downsampling, multi-target, adaptive-error, error-consistency, and
timing outputs are described above.

```
📂 fe57fcf2445a2909e688bff847585546/
├── 📂 features/
│   ├── 📂 associations/
│   │   ├── associations_report.md
│   │   ├── categorical_features.csv
│   │   ├── categorical_features.parquet
│   │   ├── continuous_features.csv
│   │   └── continuous_features.parquet
│   └── 📂 predictions/
│       ├── categorical_features.csv
│       ├── categorical_features.parquet
│       ├── continuous_features.csv
│       ├── continuous_features.parquet
│       └── predictions_report.md
├── 📂 inspection/
│   ├── inferred_types.csv
│   └── short_inspection_report.md
├── 📂 prepared/
│   ├── info.json
│   ├── idx_tests.json
│   ├── labels.parquet
│   ├── preparation_report.md
│   ├── X.parquet
│   ├── X_cat.parquet
│   ├── X_cont.parquet
│   ├── X_tabpfn.parquet
│   └── y.parquet
├── 📂 results/
│   ├── eval_htune_results_jsonpickle.json
│   ├── final_performances.csv
│   ├── performance_long_table.csv
│   ├── results_report.md
│   ├── run_timing.json
│   ├── X_test.csv
│   ├── X_train.csv
│   ├── y_test.csv
│   └── y_train.csv
├── 📂 selection/
│   ├── 📂 embed/
│   │   ├── <model>_embed_selection_data.json
│   │   └── <model>_embedded_selection_report.md
│   ├── 📂 filter/
│   │   ├── association_selection_report.md
│   │   └── prediction_selection_report.md
│   └── 📂 wrapper/
│       ├── wrapper_selection_data.json
│       └── wrapper_selection_report.md
├── 📂 tuning/
│   └── tuned_models.csv
├── options.json
└── terminal_outputs.txt
```

# Limitations

- malformed data (e.g. quoting, feature names with spaces or commas,
  malformed `.csv`, etc)
- [inappropriate data](#inappropriate-data) (e.g. timeseries or sequence
  data, NLP data)
- [inappropriate tasks](#inappropriate-tasks) (e.g. unsupervised learning
  tasks)
- dataset size: **expected max runtime should be well under 24 hours** (see
  [below](#dataset-size) for how to estimate your expected runtime on the
  Compute Canada / DRAC Niagara cluster)
- wrapper selection is extremely expensive and the number of selected
  features (or eliminated features, in the case of step-down selection) should
  *not* exceed:
    - step-up: 20
    - step-down: 10

## Target Variables

Use `--target` for one target or `--targets` for several. Features and targets
need different handling throughout the analysis. For example:

- normalization of targets in regression must be different than normalization
  of continuous features
- samples with NaNs in the target must be dropped (resulting in a different
  base dataframe), but samples with NaN features can be imputed
- data splitting must be stratified in classification to avoid errors, but
  stratification must be based on the target (e.g. choosing a different
  target will generally result in different splits)

Feature selection is run for each target, then combined using the selected
multi-target aggregation strategy. Multi-target classification uses a
support-aware split proxy and records when stratification must be relaxed.

## Dataset Size

Let $p$ be the number of features, and $n$ be the number of samples in the
tabular data. Also, let $h = 3600 = 60 \times 60$ be the number of seconds in
an hour.

Based on some experiments with about 70 datasets from the [OpenML
platform](https://www.openml.org/search?type=data&sort=runs&status=active)
and on the [Niagara compute
cluster](https://docs.alliancecan.ca/wiki/Niagara#Node_characteristics), and
limiting wrapper-based selection to step-up selection of 10 features, then a
simple rule for predicting the maximum expected runtime, $t_{\text{max}}$, in
seconds, of `df-analyze` with all options and models is:

$$ t_{\text{max}} = n + 20p + 2h $$

for $n$ less than $30 000$ and for $p$ less than $200$ or so. Doubling the
amount of selected features should probably roughly double the expected max
runtime, but this is not confirmed by experiments.

These are historical estimates for the full, dense pipeline and should not be
extrapolated to the large-scale downsampling paths. Downsampling can reduce the
feature count seen by later analyses, but the ordinary and large-feature table
paths must still load their source table in memory; use SVMlight for a source
matrix that must remain sparse.

The estimate also predates multi-target analysis, KAN, TabPFN, adaptive error,
and error consistency, and should not be used for those combinations. Error
consistency adds `folds x repetitions` refits for every target, model, and
selected feature set: 25 with the default 5 x 5 settings, 50 with the
classification profile, and 250 with the regression profile. Use a 2 x 2 run
with one model to measure the dataset and machine before scaling up.

The expected runtime on your machine will be quite different. If $n <
10 000$ and $p < 60$, and you have a recent machine (e.g. M1/M2/M3 series
Mac) then it is likely that your runtimes will be significantly faster than
the above estimate.

Also, it is extremely challenging to predict the runtimes of AutoML
approaches like `df-analyze`: besides machine characteristics, the structure
of the data (beyond just the number of samples and features) and
hyperparameter values (e.g. for [support vector
machines](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py))
also have a significant impact on fit times.

**Datasets with $n > 30 000$, i.e. over 30 000 samples, should be considered
potentially intractable for the full all-model, wrapper-selection
configuration described above**. Such a configuration is unlikely to complete
in under 24 hours, and may in fact cause out-of-memory errors (the average
Niagara node has only about 190GB of RAM, and to use all 40 cores, the dataset
must be copied 40 times due to Python's inability to properly share memory).

If you have a dataset where the expected runtime is getting close to 24 hours,
then you should strongly consider limiting the `df-analyze` options such that:

- only 2-3 models (avoiding compute-heavy `mlp`, `kan`, or `tabpfn` runs unless
  they are specifically needed) are used, AND
- wrapper-based feature selection is not used



## Inappropriate Data

Datasets with any kind of strong spatio-temporal clustering, or
spatio-temporal autocorrelation, `df-analyze` *technically* can handle (i.e.
will produce results for), but the reported results will be deeply invalid
and misleading. This includes:

- **time-series or sequence data**, especially where the task is
  *forecasting*,
  - This means data where the target variable is either a categorical or
    continuous variable that represents some subsequent or future state of a
    sequence of samples in the training data, e.g. predicting weather, stock
    prices, media popularity, and etc., but where a correct predictive model
    necessarily must know recent target values
  - naive k-fold splitting is [completely
    invalid](https://stats.stackexchange.com/a/14109) in these kinds of
    cases, and k-fold is the basis of most of the main analyses in
    `df-analyze`

- **spatially autocorrelated** or **autocorrelated data** in general
  - A generalization of the case above, but the same problem: k-fold is
    invalid when [the similarity of samples that are close in space is not
    accounted for in splitting](https://arxiv.org/abs/2005.14263)

- **unencoded text data / natural language processing (NLP) data**
  - I.e. any data where a sample feature is a word or collection of words

- **image data** (e.g. computer vision prediction tasks)
  - These datasets will almost always be too expensive for the ML algorithms
    in `df-analyze` to process directly; use `df-embed.py` to convert supported
    image data to tabular embeddings, or use a task-specific vision model

## Inappropriate Tasks

Anything beyond simple prediction, e.g. unsupervised tasks like clustering,
representation learning, dimension reduction, or even semi-supervised tasks,
are simply beyond the scope of `df-analyze`.


# Currently Implemented Program Features and Analyses

## Completed Features

### Automated Data Preprocessing

- **NaN Removal and Handling**
  - samples with NaN target are dropped (see Target Handling below)
  - continuous-feature NaNs are median-imputed using the training partition
  - the legacy `--nan` choices are still accepted for configuration
    compatibility, but the current preparation paths use median imputation
  - categorical features encode NaNs as an additional class / level

- **Bad Feature Detection and Removal**
  - features containing unusable datetime data (e.g. timeseries data) are
    automatically detected and removed, with warnings to the user
  - features containing identifiers (e.g. features that are integer or string
    and where each sample has a unique value) are automatically detected and
    removed, with user warnings
  - extremely large categorical features (more categories than about 1/5 of
    samples, which pose a problem for 5-fold splitting) are automatically
    removed with user-warnings
  - "suspicious" integer features (e.g. features with less than 5 unique
    values) are detected and the user is warned to check if categorical or
    ordinal

- **Categorical Feature Handling**
  - user can
    - specify categorical feature names explicitly (preferred)
    - specify a threshold (count) for number of unique values of a feature
      required to count as categorical
  - string features (even if not user-specified) are one-hot encoded
  - NaN values are automatically treated as an additional class level (no
    dropping of NaN samples required)

- **Continuous Feature Handling**
  - continuous features use robust normalization by default
  - the legacy `--norm` choices `minmax` and `robust` are accepted for
    configuration compatibility, but the current preparation paths use robust
    normalization
  - normalization is fitted on training data and then applied to holdout data;
    users should still inspect extreme values and distribution shift

- **Target Handling**
  - all samples with NaN targets are dropped (categorical or continuous)
    - rarely makes sense to count correct NaN predictions toward classification
      performance
    - imputing NaNs in a regression target (e.g. mean, median) biases estimates
      of regression performance
  - categorical targets containing a class with 20 or fewer samples in a level
    have the samples corresponding to that level dropped, and the user is
    warned (these cause problems in nested stratified k-fold, and any estimated
    of any metric or performance on such a small class is essentially
    meaningless)
  - continuous or ordinal single regression targets are robustly normalized
    using 2.5th and 97.5th percentile values
    - with this normalization, 95% of the target values are in [0, 1]
    - thus an MAE of, say, 0.5, means that the error is about half of the
      target (robust) range
    - this also aids in the convergence and fitting of scale-sensitive models
    - this also makes prediction metrics (e.g. MAE) more comparable across
      different targets
  - for multi-target classification, low-support levels are retained so valid
    labels in the other targets are not silently discarded; the preparation
    report records them and the user is warned
  - before model tuning, every level in each multi-target training partition
    must support every selected K-fold model's declared tuning design (currently
    three or five folds); non-K-fold tuners are excluded from this specific
    preflight. Holdout support is checked against the resolved final-CV fold
    count (up to five, and limited by group count); otherwise the run stops with
    the target, level, observed count, and required count
  - an internally generated classification holdout must contain every level
    present in training. An externally supplied holdout may omit a training
    level; when its remaining levels still satisfy fold support, the run emits
    an explicit warning because the omitted level's performance cannot be
    estimated
  - binary sensitivity, specificity, PPV and NPV use encoded class `1`; each
    per-target result table and Markdown report records the corresponding
    original positive-class label
  - each actual multi-target tuning and final-CV fold is also verified after
    splitting: classification targets retain the required per-level support,
    and every regression target varies in both sides of every fold.
    Deterministic alternative partitions are attempted before an impossible
    grouped or ungrouped design is rejected
  - multi-target regression targets remain in their original units in saved
    predictions and per-target MAE/MSE reports. Model fitting uses independent
    training-only target standardization and then inverse-transforms
    predictions
  - raw errors from targets with different units are not averaged for model
    ranking. Aggregate regression output reports normalized `multi-nmae`,
    `multi-nmse`, and `multi-nrmse`, plus scale-independent R²; explicitly
    prefixed `raw-*` metrics are informational only
  - every regression target must vary in both the outer training and holdout
    partitions; deterministic alternative outer partitions are tried before an
    impossible design is rejected



### Feature Descriptive Statistics

- **Continuous and ordinal features**:
  - Non-robust:
    - min, mean, max, standard deviation (SD)
  - Robust:
    - 5th and 95th percentiles, median, interquartile range (IQR)
  - Moments/Other:
    - skew, kurtosis, and p-values that skew/kurtosis differ from Gaussian
    - entropy (e.g. differential / continuous entropy)
    - NaN counts and frequency

- **Categorical features**:
  - number of classes / levels
  - min, max, and median of class frequencies
  - heterogeneity (Chi-squared test of equal class sizes) and associated p-value
  - NaN counts and frequency (treated as another class label)

### Univariate Feature-Target Associations

- **Continuous/Ordinal Feature -> Categorical Target**:
  - Statistical: t-test, Mann-Whitney U, Brunner-Munzel W, Pearson r and
    associated p-values
  - Other: Cohen's d, AUROC, mutual information
- **Continuous/Ordinal Feature -> Continuous Target**:
  - Pearson's and Spearman's r and p-values
  - [F-test](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
    and p-value
  - mutual information

- **Categorical Feature -> Continuous Target**:
  - [Kruskal-Wallis
    H](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)
    and p-value
  - mutual information
  - **NOTE**: There are relatively few measures of association for
    categorical-continuous variable pairs. Kruskal-Wallis H has few
    statistical assumptions, and essentially checks the extent that the medians
    of each level in the categorical variable differ significantly on the
    continuous target.

- **Categorical Feature -> Categorical Target**:
  - Cramer's V

### Univariate Prediction Metrics for each Feature-Target Pair

- simple linear predictive models (sklearn
  [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
  and
  [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html))
  are tuned with 5-fold cross-validation over a small grid for each feature
- a dummy regressor or classifier (e.g. predict target mean, predict largest
  class) is also always fit
- reported metrics are for the best-tuned model mean performance across the 5
  folds:
  - **Categorical Target (e.g. classification)**:
    - Models: DummyClassifier and SGDClassifier
    - Metrics: accuracy, AUROC, sensitivity, specificity, F1, balanced accuracy
  - **Continuous/Ordinal Target (e.g. regression)**:
    - Models: DummyRegressor and SGDRegressor
    - Metrics: mean absolute error, mean squared error, median absolute error, R²
