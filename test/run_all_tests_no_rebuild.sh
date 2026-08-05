#!/bin/bash

set -euo pipefail

# Only intended for use on MacOS and/or Linux local install
TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
echo "$ROOT"
cd "$ROOT" || exit 1

if [[ -z "${CC_CLUSTER:-}" ]]; then
    echo "On local machine, will use virtual environment for testing"
    VENV="$ROOT/.venv"
    PYTEST="$VENV/bin/pytest"
    PYTHON="$VENV/bin/python"
else
    # shellcheck disable=SC2016
    echo 'On Compute Canada, will use container-defined $PYTEST variable'
    module load apptainer
    export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
    export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
    apptainer run --home "$(readlink -f .)" df_analyze.sif "$(readlink -f test/cc_run_all_tests_no_rebuild.sh)"
    exit 0
fi


echo "================================================================================="
echo "Testing basic data inspection, cleaning, preparation, and associational stats"
echo "================================================================================="
"$PYTEST" \
    -m 'not regen and fast' -x \
    test/test_inspection.py \
    test/test_prepare.py \
    test/test_splitting.py \
    test/test_seeding.py \
    test/test_associate.py \
    test/test_name_sanitize.py \
    test/test_cleaning.py || { echo "Basic functionality failed."; exit 1; }

echo "================================================================================="
echo "Testing basic CLI functionality"
echo "================================================================================="
"$PYTEST" \
    test/test_cli_parsing.py \
    test/test_cli_random.py \
    -x || { echo "CLI testing failed."; exit 1; }

echo "================================================================================="
echo "Testing error consistency, repeated K-fold, and EC CLI integration"
echo "================================================================================="
"$PYTEST" \
    test/test_error_consistency.py \
    test/test_error_consistency_cli_e2e.py \
    -x || { echo "Error-consistency testing failed."; exit 1; }

echo "================================================================================="
echo "Testing added models, devices, downsampling, multi-target, and runtime setup"
echo "================================================================================="
"$PYTEST" \
    test/test_added_models.py \
    test/test_device.py \
    test/test_downsampling.py \
    test/test_lgbm_defaults.py \
    test/test_multitarget.py \
    test/test_preprocessing_fit_scope.py \
    test/test_real_gpu_integrations.py \
    test/test_runtime_install.py \
    test/test_sparse_downsampling.py \
    -x || { echo "New feature testing failed."; exit 1; }

echo "================================================================================="
echo "Testing test dataset IO"
echo "================================================================================="
"$PYTEST" -n auto \
    test/test_loading.py \
    test/test_datasets.py -x || { echo "Basic Dataset IO failed."; exit 1; }

echo "================================================================================="
echo "Sanity testing basic models and scoring"
echo "================================================================================="
"$PYTEST" -n 4 \
    test/test_models.py || { echo "Model sanity test(s) failed."; exit 1; }
"$PYTEST" -n auto \
    test/test_tuning_score.py -x || { echo "IO with tuning scores failed."; exit 1; }
"$PYTEST" -n auto \
    test/test_confusion_mat_scoring.py -x || { echo "Confusion matrix scoring failed."; exit 1; }

echo "================================================================================="
echo "Testing result saving (slow)"
echo "================================================================================="
"$PYTEST" -n auto test/test_saving.py -x || { echo "Saving failed."; exit 1; }

echo "================================================================================="
echo "Testing prediction"
echo "================================================================================="
"$PYTEST" test/test_predict.py -m 'fast' -x || { echo "Prediction testing failed."; exit 1; }

echo "================================================================================="
echo "Testing selection. This is very slow, and if the first 5-10 pass, the rest "
echo "likely will, so feel free to Ctrl+C or Ctrl+Z at this point..."
echo "================================================================================="
"$PYTEST" test/test_selection.py -m 'fast' -x || { echo "Selection testing failed."; exit 1; }

echo "================================================================================="
echo "Testing embeddding functionality."
echo "================================================================================="
"$PYTHON" df-embed.py --modality nlp --download  || { echo "Failed to download NLP model";  exit 1; }
"$PYTHON" df-embed.py --modality vision --download || { echo "Failed to download vision model"; exit 1; }
# The faster tests also test the same things the slow ones do, so no need to
# run them here. They can always be run manually to investigate issues.
"$PYTEST" test/test_embedding.py -m 'fast' || { echo "Failed to perform embedding."; exit 1; }


# # Done
# test_inspection.py
# test_prepare.py
# test_associate.py
# test_cleaning.py
# test_loading.py
# test_cli_random.py
# test_cli_parsing.py
# test_error_consistency.py
# test_error_consistency_cli_e2e.py
# test_datasets.py
# test_saving.py
# test_models.py
# test_predict.py
# test_name_sanitize.py
# test_tuning_score.py

# # Integration
# test_main.py
# test_spreadsheets.py

# # ERRORS!
# test_mlp.py  # segmentation fault


## TODO
# test_selection.py
