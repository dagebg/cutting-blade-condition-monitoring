"""
Preprocessing utilities for the cutting blade condition monitoring project.

This module contains reusable functions that clean and prepare raw time-series
sensor data from the OCME Vega shrink-wrap machine before feature engineering.

Responsibilities include:
- Loading and validating per-file DataFrame structures.
- Dropping irrelevant or redundant signals.
- Checking for missing values and duplicate timestamps.
- Applying lightweight, file-level annotations needed for later processing.

The focus here is on keeping the raw sensor data consistent and reliable,
while leaving feature and label construction to `features.py`.
"""

from pathlib import Path
import pandas as pd


def drop_irrelevant_signals_inplace(df_list: list[pd.DataFrame]):
    """
    Drop irrelevant pSintor::VAX_speed columns from the list of DataFrames in place.

    Args:
        df_list (list): list of pandas DataFrames.
    """
    for df in df_list:
        if "pSpintor::VAX_speed" in df.columns:
            df.drop(columns=["pSpintor::VAX_speed"], inplace=True)


def check_missing_and_duplicates(
    df_list: list[pd.DataFrame], data_files: list[Path], whitelist_files: set[str]
):
    """Checks for missing rows and duplicate rows in the list of DataFrames.
        Ignores the whiteliisted files. Prints the filenames and the number
        of missing values and duplicate rows for each file or prints a message if none were found.

    Args:
        df_list (list[pd.DataFrame]): list containing the DataFrames to check.
        data_files (list[Path]): list containing the paths of the DataFrames.
        whitelist_files (set[str]): set containing the filenames to ignore.
    """
    problems_found = (
        False  # Flag: stays False unless we detect any "real" issue in any file
    )

    for i, df in enumerate(df_list):
        filename = data_files[
            i
        ].name  # Get the corresponding filename for this DataFrame

        nan_count = (
            df.isna().sum().sum()
        )  # Total number of NaN cells in the whole DataFrame
        duplicate_count = (
            df.duplicated().sum()
        )  # Number of duplicated rows (full-row duplicates)

        is_whitelisted = filename in whitelist_files

        # A file is considered problematic if:
        # - it has NaNs AND is NOT whitelisted, OR
        # - it has duplicate rows (duplicates are always considered problems here)
        has_real_problems = (not is_whitelisted and nan_count > 0) or (
            duplicate_count > 0
        )

        if has_real_problems:
            print(f"File: {filename}")

            # Only report NaNs if the file is not on the whitelist
            if not is_whitelisted and nan_count > 0:
                print(f"Number of missing values (NaN): {nan_count}")

            # Always report duplicates if any exist
            if duplicate_count > 0:
                print(f"Number of duplicate rows: {duplicate_count}")

            print("-" * 40)
            problems_found = True  # At least one problem was found across the dataset

    # If the loop finished and we never flagged a problem, print the all-clear message
    if not problems_found:
        print("No missing values or duplicate rows were found in any file.")
