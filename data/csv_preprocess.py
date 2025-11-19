"""Utility helpers for CSV-based preprocessing pipelines.

This module provides a reusable command-line interface for preparing
custom datasets before feeding them into OneNet. It covers three
common cleaning steps:

1. Selecting a subset of columns that will be used as model features.
2. Filling missing values with statistics derived from the present data.
3. Filtering obvious outliers with a z-score threshold.
4. Optionally renaming columns to harmonise feature naming conventions.
5. Resampling the dataset to a new temporal frequency before cleaning.

Example
-------
Use the script to prepare ``raw.csv`` and keep the columns ``load`` and
``temperature`` as features while keeping ``load`` as the prediction
target::

    python utils/csv_preprocess.py \
        --input raw.csv \
        --output cleaned.csv \
        --features load temperature \
        --target load \
        --fill-method median \
        --z-threshold 3.5 \
        --rename load:load_kw temperature:temp_c \
        --datetime-col timestamp \
        --resample-freq 1H

The resulting ``cleaned.csv`` can be consumed by the existing data
loaders in OneNet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for CSV preprocessing."""

    input_path: Path
    output_path: Path
    feature_columns: Sequence[str]
    target_column: str | None = None
    fill_method: str = "median"
    z_threshold: float = 3.0
    rename_map: Mapping[str, str] | None = None
    datetime_column: str | None = None
    resample_freq: str | None = None


def _resolve_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Columns not found in input file: " + ", ".join(missing)
        )
    return list(columns)


def _fill_missing_values(df: pd.DataFrame, method: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    if method == "median":
        fill_values = df[numeric_cols].median(numeric_only=True)
    elif method == "mean":
        fill_values = df[numeric_cols].mean(numeric_only=True)
    elif method == "zero":
        fill_values = pd.Series(0.0, index=numeric_cols)
    else:
        raise ValueError(
            "Unsupported fill method: {method}. Choose from 'median', 'mean', 'zero'."
            .format(method=method)
        )

    df.loc[:, numeric_cols] = df[numeric_cols].fillna(fill_values)

    for col in categorical_cols:
        if df[col].isna().all():
            continue
        mode_value = df[col].mode(dropna=True)
        if not mode_value.empty:
            df.loc[:, col] = df[col].fillna(mode_value.iloc[0])

    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def _filter_outliers(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    if z_threshold <= 0:
        return df

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return df

    mean = numeric_df.mean()
    std = numeric_df.std(ddof=0)
    std_replaced = std.replace(0, np.nan)
    z_scores = (numeric_df - mean) / std_replaced
    mask = z_scores.abs() <= z_threshold
    mask = mask | mask.isna()
    keep_rows = mask.all(axis=1)
    return df.loc[keep_rows].reset_index(drop=True)


def preprocess_csv(config: PreprocessConfig) -> pd.DataFrame:
    """Run the preprocessing pipeline and return the cleaned frame."""

    df = pd.read_csv(config.input_path)

    rename_map = dict(config.rename_map or {})
    if rename_map:
        missing = [col for col in rename_map if col not in df.columns]
        if missing:
            raise ValueError(
                "Columns not found for renaming: " + ", ".join(missing)
            )
        df = df.rename(columns=rename_map)

    if config.resample_freq:
        if not config.datetime_column:
            raise ValueError(
                "Resampling requires --datetime-col to specify the timestamp column."
            )

        datetime_col = rename_map.get(
            config.datetime_column, config.datetime_column
        )
        if datetime_col not in df.columns:
            raise ValueError(
                f"Datetime column '{datetime_col}' not found in the input file."
            )

        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        if df[datetime_col].isna().all():
            raise ValueError(
                f"Datetime column '{datetime_col}' could not be parsed to datetime."
            )

        df = df.sort_values(datetime_col)
        df = df.set_index(datetime_col)

        agg_map = {
            col: "mean"
            for col in df.columns
            if col != datetime_col and pd.api.types.is_numeric_dtype(df[col])
        }
        non_numeric_cols = [
            col
            for col in df.columns
            if col != datetime_col and col not in agg_map
        ]
        if non_numeric_cols:
            raise ValueError(
                "Resampling currently supports numeric columns only. Non-numeric "
                "columns present: " + ", ".join(non_numeric_cols)
            )

        df = df.resample(config.resample_freq).agg(agg_map)
        df = df.reset_index()

    feature_columns = [rename_map.get(col, col) for col in config.feature_columns]
    selected_columns: List[str] = _resolve_columns(df, feature_columns)

    target_column = (
        rename_map.get(config.target_column, config.target_column)
        if config.target_column
        else None
    )
    if target_column and target_column not in selected_columns:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in the input file."
            )
        selected_columns.append(target_column)

    if config.resample_freq:
        datetime_col = rename_map.get(
            config.datetime_column, config.datetime_column
        )
        if datetime_col not in selected_columns:
            selected_columns = [datetime_col] + selected_columns

    df = df[selected_columns].copy()
    df = _fill_missing_values(df, config.fill_method)
    df = _filter_outliers(df, config.z_threshold)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_path, index=False)
    return df


def _parse_args(argv: Iterable[str] | None = None) -> PreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Preprocess a CSV file for OneNet experiments."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the source CSV file.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to save the cleaned CSV file.",
    )
    parser.add_argument(
        "--features",
        required=True,
        nargs="+",
        help="Columns to keep as model features.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column to append to the output.",
    )
    parser.add_argument(
        "--fill-method",
        choices=("median", "mean", "zero"),
        default="median",
        help="How to fill missing numeric values.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for outlier filtering (set <= 0 to disable).",
    )
    parser.add_argument(
        "--rename",
        nargs="+",
        metavar="OLD:NEW",
        default=(),
        help=(
            "Optional column rename directives. Use OLD:NEW syntax and repeat "
            "for multiple columns."
        ),
    )
    parser.add_argument(
        "--datetime-col",
        help="Name of the timestamp column used for resampling (required when --resample-freq is set).",
    )
    parser.add_argument(
        "--resample-freq",
        help="Optional pandas frequency string for time-based resampling before cleaning.",
    )
    args = parser.parse_args(argv)

    rename_map: dict[str, str] = {}
    for item in args.rename:
        if ":" not in item:
            parser.error(
                "Invalid --rename argument '{item}'. Expected format OLD:NEW.".format(
                    item=item
                )
            )
        old, new = item.split(":", 1)
        if not old or not new:
            parser.error(
                "Invalid --rename argument '{item}'. Neither OLD nor NEW can be empty.".format(
                    item=item
                )
            )
        if old in rename_map:
            parser.error(
                "Duplicate rename directive for column '{col}'.".format(col=old)
            )
        rename_map[old] = new

    return PreprocessConfig(
        input_path=args.input,
        output_path=args.output,
        feature_columns=tuple(args.features),
        target_column=args.target,
        fill_method=args.fill_method,
        z_threshold=args.z_threshold,
        rename_map=rename_map or None,
        datetime_column=args.datetime_col,
        resample_freq=args.resample_freq,
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = _parse_args(argv)
    preprocess_csv(config)


if __name__ == "__main__":
    main()