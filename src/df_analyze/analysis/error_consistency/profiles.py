"""Named EC reference profiles and explicit-override handling.

The profiles set folds, repetitions, holdout size, and model-seed behaviour.
They do not reproduce a paper's full data and modelling pipeline.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any

REGRESSION_PAPER_METHODS = (
    "ratio",
    "ratio_diff",
    "ratio_sign",
    "ratio_diff_sign_reference",
    "intersection_union_sample",
    "intersection_union_all",
    "intersection_union_distance",
)


@dataclass(frozen=True)
class ECProfile:
    name: str
    problem_type: str | None
    settings: dict[str, Any]
    scope: str


EC_PROFILES: dict[str, ECProfile] = {
    "none": ECProfile(
        name="none",
        problem_type=None,
        settings={},
        scope="df-analyze defaults",
    ),
    "classification-paper": ECProfile(
        name="classification-paper",
        problem_type="classification",
        settings={
            "test_val_size": 0.2,
            "ec_folds": 5,
            "ec_repetitions": 10,
            "ec_model_seed_mode": "fixed",
            "ec_holdout_role": "test",
        },
        scope=(
            "Classification EC reference settings: repeated K-fold fits on one "
            "shared holdout. The original data and model pipeline are not included."
        ),
    ),
    "regression-paper": ECProfile(
        name="regression-paper",
        problem_type="regression",
        settings={
            "test_val_size": 0.2,
            "ec_folds": 5,
            "ec_repetitions": 50,
            "ec_model_seed_mode": "fixed",
            "ec_methods": REGRESSION_PAPER_METHODS,
            "ec_holdout_role": "test",
        },
        scope=(
            "Regression EC reference settings for real-world tabular experiments, "
            "including the signed ratio-diff-sign method. The original preprocessing "
            "and models are not included."
        ),
    ),
}

EC_PROFILE_CHOICES = tuple(EC_PROFILES)

GENERAL_EC_DEFAULTS: dict[str, Any] = {
    "test_val_size": 0.4,
    "ec_folds": 5,
    "ec_repetitions": 5,
    "ec_model_seed_mode": "vary",
    "ec_methods": None,
    "ec_holdout_role": "test",
}


def apply_ec_profile(cli_args: Namespace, mode: str) -> Namespace:
    """Resolve a paper profile while preserving explicitly supplied values.

    Profile-controlled parser arguments use ``None`` as their unresolved default.
    Consequently, a CLI or spreadsheet value always wins, including when the
    explicit value happens to equal the ordinary df-analyze default.
    """

    name = str(getattr(cli_args, "ec_profile", "none") or "none").lower()
    profile = EC_PROFILES[name]
    problem_type = "classification" if "class" in str(mode).lower() else "regression"
    if profile.problem_type is not None and profile.problem_type != problem_type:
        raise ValueError(
            f"EC profile '{name}' requires {profile.problem_type} mode, "
            f"but mode '{mode}' was requested."
        )

    overrides: dict[str, Any] = {}
    for key, profile_value in profile.settings.items():
        current = getattr(cli_args, key, None)
        if current is None:
            setattr(cli_args, key, profile_value)
        elif current != profile_value:
            overrides[key] = current

    for key, default in GENERAL_EC_DEFAULTS.items():
        if getattr(cli_args, key, None) is None:
            setattr(cli_args, key, default)

    if name != "none":
        cli_args.error_consistency = True
    cli_args.ec_profile = name
    cli_args.ec_profile_scope = profile.scope
    cli_args.ec_profile_overrides = overrides
    return cli_args
