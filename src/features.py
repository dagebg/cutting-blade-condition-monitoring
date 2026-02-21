"""
Feature engineering utilities for the cutting blade condition monitoring project.

This module encapsulates all logic that transforms raw time-series sensor data
from the OCME Vega shrink-wrap machine into aggregated, tabular features and
labels that can be used for downstream modeling tasks.

Responsibilities include:
- Parsing filenames to derive degradation and operating-mode labels.
- Adding heuristic anomaly labels based on controller lag error signals.
- Computing statistical and frequency-domain features from selected signals.
- Building a consolidated feature table from all raw CSV files and saving it.

The goal is to keep notebooks lightweight by centralizing reusable feature
engineering code in one place, ensuring consistency and easier experimentation.
"""

from datetime import datetime
import logging
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)


def extract_degradation_label(file_obj):
    """
    Extracts a degradation label (0–1) from a file object whose .name
    starts with a timestamp like 'MM-DDThhmmss_...'.

    Assumes year 2024 (leap year), which covers all possible dates and is
    sufficient because only the relative temporal order matters.

    Args:
        file_obj (Path): file object to extract the label from.

    Returns:
        normalized day of year as float between 0 and 1 or NaN if an error occurred.
    """
    date_str = file_obj.name.split("_")[0]
    try:
        date_obj = datetime.strptime(f"2024-{date_str}", "%Y-%m-%dT%H%M%S")
        day_of_year = date_obj.timetuple().tm_yday
        return day_of_year / 366.0
    except Exception as e:
        logger.warning(
            "Error processing file %s: %s - Filling with NaN", file_obj.name, e
        )
    return np.nan


def add_degradation_labels(df_list: list[pd.DataFrame]):
    """Adds a 'degradation' column to each DataFrame in df_list based on the
    corresponding entry in data_files, and returns df_list sorted by
    the degradation value of the first row (NaNs go to the end).

    Args:
        df_list (list[pd.DataFrame]): list of DataFrames to add the labels to.
        data_files (list[Path]): list of file paths corresponding to the DataFrames.

    Returns:
          df_list_sorted (list[pd.DataFrame]): sorted list of DataFrames with added labels.
    """

    for df in df_list:
        file_path = Path(df["filename"].iloc[0])
        df["degradation"] = extract_degradation_label(file_path)

    df_list_sorted = sorted(
        df_list,
        key=lambda d: (
            d["degradation"].iloc[0]
            if not pd.isna(d["degradation"].iloc[0])
            else float("inf")  # NaNs go to the end
        ),
    )

    return df_list_sorted


def extract_mode_label_from_filename(filename: str) -> str:
    """Extract the operating mode label (e.g. 'mode1', 'mode2') from a CSV filename.

    Assumes filenames follow the pattern '<timestamp>_<index><mode>.csv',
    for example: '01-04T184148000mode1.csv'.

    Args:
        filename (str): Name of the CSV file.

    Returns:
        str: Extracted mode label (e.g. 'mode1').
    """
    return filename.split("_")[-1].replace(".csv", "")


def add_mode_labels(
    df_list: list[pd.DataFrame],
    column_name: str = "mode",
) -> list[pd.DataFrame]:
    """Add a mode column to each DataFrame based on its internal 'filename' column.

    The function extracts the operating mode (e.g. 'mode1', 'mode2') from the
    'filename' column already present in each DataFrame and adds it as a new
    column (default: 'mode'). This approach is safe regardless of the order
    df_list has been sorted into.

    Args:
        df_list (list[pd.DataFrame]): List of DataFrames to add the labels to.
            Each DataFrame must contain a 'filename' column.
        column_name (str): Name of the mode column to add. Defaults to "mode".

    Returns:
        list[pd.DataFrame]: The input list with the new mode column added in place.
    """
    for df in df_list:
        df[column_name] = extract_mode_label_from_filename(df["filename"].iloc[0])

    return df_list


def add_heuristic_anomaly_labels(
    df_list: list[pd.DataFrame],
    lag_error_col: str = "pCut::CTRL_Position_controller::Lag_error",
    std_factor: float = 1.5,
    anomaly_ratio_threshold_per_file: float = 0.05,
) -> list[pd.DataFrame]:
    """Adds heuristic anomaly labels at row and file level to each DataFrame.

    The procedure:
      1. For each file, compute mean and standard deviation of `Lag_error`.
      2. Define the threshold:
         Anomaly = |Lag_error| > (mean + std_factor × standard deviation).
      3. Label each row with `anomaly` (0/1) based on this rule.
      4. Label the whole file with `file_anomaly` (0/1) if more than
         `anomaly_ratio_threshold_per_file` (e.g. 0.05) of the rows are anomalous.

    Args:
        df_list (list[pd.DataFrame]): List of DataFrames to label.
        lag_error_col (str): Name of the Lag_error signal column.
        std_factor (float): Factor for the standard deviation in the threshold.
        anomaly_ratio_threshold_per_file (float): Minimum anomalous row ratio
            per file to mark the file as anomalous.

    Returns:
        list[pd.DataFrame]: The input list with added `anomaly` and
            `file_anomaly` columns.
    """
    for df in df_list:
        mean = df[lag_error_col].mean()
        std = df[lag_error_col].std()

        threshold = mean + std_factor * std

        df["anomaly"] = (df[lag_error_col].abs() > threshold).astype(int)
        df["file_anomaly"] = int(
            df["anomaly"].mean() > anomaly_ratio_threshold_per_file
        )

    return df_list


def add_fft_lag_error_features(
    df_list: List[pd.DataFrame],
    lag_error_col: str = "pCut::CTRL_Position_controller::Lag_error",
    sampling_rate: float = 250.0,
) -> List[pd.DataFrame]:
    """
    Compute simple FFT-based features for the Lag_error signal in each file.

    For each DataFrame in `df_list`, this function computes:
      - fft_max_amplitude: maximum amplitude in the magnitude spectrum
      - fft_dominant_freq: frequency with maximum amplitude
      - fft_total_energy: sum of squared spectral magnitudes

    These features are written as constant columns per file.
    """
    for df in df_list:
        signal = df[lag_error_col].values
        N = len(signal)

        fft_vals = np.abs(rfft(signal))
        fft_freqs = rfftfreq(N, d=1 / sampling_rate)

        # extracting features
        max_amplitude = np.max(fft_vals)
        freq_max_amp = fft_freqs[np.argmax(fft_vals)]
        total_energy = np.sum(fft_vals**2)

        # writing features to each file
        df["fft_max_amplitude"] = max_amplitude
        df["fft_dominant_freq"] = freq_max_amp
        df["fft_total_energy"] = total_energy

    return df_list


SELECTED_SIGNALS = [
    "pCut::CTRL_Position_controller::Lag_error",
    "pCut::CTRL_Position_controller::Actual_speed",
    "pCut::CTRL_Position_controller::Actual_position",
    "pCut::Motor_Torque",
    "pSvolFilm::CTRL_Position_controller::Lag_error",
    "pSvolFilm::CTRL_Position_controller::Actual_speed",
    "pSvolFilm::CTRL_Position_controller::Actual_position",
]


def build_feature_table(
    df_list: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate per-file statistical features and targets into a feature matrix.

    For each file, the function computes mean, std, min, max and outlier_ratio
    (|z| > 2) for all SELECTED_SIGNALS and appends mode, degradation and
    file_anomaly as target columns.
    """
    feature_vectors = []

    for df in df_list:
        features: dict[str, float] = {}

        for signal in SELECTED_SIGNALS:
            values = df[signal].values
            mean = np.mean(values)
            std = np.std(values)

            features[f"{signal}__mean"] = mean
            features[f"{signal}__std"] = std
            features[f"{signal}__min"] = np.min(values)
            features[f"{signal}__max"] = np.max(values)
            features[f"{signal}__outlier_ratio"] = np.mean(
                np.abs((values - mean) / (std + 1e-8)) > 2
            )

        # targets
        features["mode"] = df["mode"].iloc[0]
        features["degradation"] = df["degradation"].iloc[0]
        features["file_anomaly"] = df["file_anomaly"].iloc[0]

        feature_vectors.append(features)

    feature_df = pd.DataFrame(feature_vectors)

    # optional: rename to short, readable names
    feature_df.columns = [
        "K_PF_mean",
        "K_PF_std",
        "K_PF_min",
        "K_PF_max",
        "K_PF_outl",
        "K_SPD_mean",
        "K_SPD_std",
        "K_SPD_min",
        "K_SPD_max",
        "K_SPD_outl",
        "K_POS_mean",
        "K_POS_std",
        "K_POS_min",
        "K_POS_max",
        "K_POS_outl",
        "K_TOR_mean",
        "K_TOR_std",
        "K_TOR_min",
        "K_TOR_max",
        "K_TOR_outl",
        "F_PF_mean",
        "F_PF_std",
        "F_PF_min",
        "F_PF_max",
        "F_PF_outl",
        "F_SPD_mean",
        "F_SPD_std",
        "F_SPD_min",
        "F_SPD_max",
        "F_SPD_outl",
        "F_POS_mean",
        "F_POS_std",
        "F_POS_min",
        "F_POS_max",
        "F_POS_outl",
        "mode",
        "degradation",
        "file_anomaly",
    ]

    # sort by degradation ascending (NaNs at the end)
    feature_df = feature_df.sort_values(
        "degradation", ascending=True, na_position="last"
    ).reset_index(drop=True)

    return feature_df
