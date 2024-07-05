import os
import ast
import argparse

import pandas as pd
import numpy as np

from config import GENERAL


def convert_to_float(values):
    """
    Convert a list of string values to float, handling 'null' values.

    Parameters:
    - values (list): List of string values.

    Returns:
    - list: List of float values, 'null' converted to None.
    """
    return [float(x) if x != "null" else None for x in values]


def prepare_df(df):
    """
    Prepare the dataframe for anomaly detection.

    Parameters:
    - df (DataFrame): Input dataframe with 'keys', 'values', 'ts', and '__insert_ts' columns.

    Returns:
    - df_sorted (DataFrame): Processed dataframe sorted by 'configuration_item_id' and 'ts'.
    """
    for k in ["keys", "values"]:
        df[k] = df[k].apply(lambda i: ast.literal_eval(i)[0])

    df["values"] = df["values"].apply(convert_to_float)
    df["ts"] = pd.to_datetime(df["ts"])

    expanded_df = pd.concat(
        [pd.DataFrame([row["values"]], columns=row["keys"]) for _, row in df.iterrows()]
    )
    expanded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, expanded_df], axis=1)
    df.drop(columns=["keys", "values", "__insert_ts"], inplace=True)
    df_sorted = df.sort_values(by=["configuration_item_id", "ts"]).reset_index(
        drop=True
    )

    return df_sorted


def calculate_ema(data, alpha):
    """
    Calculate Exponential Moving Average (EMA) of a given data series.

    Parameters:
    - data (array-like): The input time series data.
    - alpha (float): Smoothing factor, between 0 and 1.

    Returns:
    - ema (ndarray): EMA values of the input data.
    """
    ema = np.zeros_like(data)
    ema[0] = next(item for item in data if not np.isnan(item))
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def detect_anomalies_ema(time_series_data, alpha, n_std):
    """
    Detect anomalies in a time series using Exponential Moving Average (EMA).

    Anomalies are detected based on residuals (difference between data and EMA)
    exceeding a threshold number of standard deviations.

    Parameters:
    - time_series_data (array-like): The input time series data.
    - alpha (float): Smoothing factor for EMA, between 0 and 1.
    - n_std (float): Number of standard deviations for anomaly detection.

    Returns:
    - anomalies (ndarray of bool): Boolean array indicating anomalies.
    """
    ema = calculate_ema(time_series_data, alpha)
    residuals = time_series_data - ema
    std_dev = np.nanstd(residuals)  # standard deviation of residuals, ignoring NaNs

    anomalies = np.zeros_like(time_series_data, dtype=bool).astype(int)

    for i in range(len(time_series_data)):
        if np.isnan(time_series_data[i]) or np.abs(residuals[i]) > n_std * std_dev:
            anomalies[i] = 1

    return anomalies


def main(
    data_dir: str = GENERAL["data_dir"],
    train_filename: str = GENERAL["train_filename"],
    test_filename: str = GENERAL["test_filename"],
):
    train_df = pd.read_csv(os.path.join(data_dir, train_filename))
    test_df = pd.read_csv(os.path.join(data_dir, test_filename))

    df = pd.concat([train_df, test_df])
    df.reset_index(drop=True, inplace=True)
    df = prepare_df(df)

    all_features = [k for k in df.columns if k.startswith(GENERAL["features_start"])]

    alpha = GENERAL["EMA_alpha"]
    n_std = GENERAL["EMA_n_std"]

    all_anomalies = {}
    config_ids = df.configuration_item_id.unique()

    for config_id in config_ids:
        all_anomalies[config_id] = {}
        config_df = df[df.configuration_item_id == config_id]

        for feature in all_features:
            time_series_data = config_df[feature].values
            anomalies = detect_anomalies_ema(time_series_data, alpha, n_std)
            all_anomalies[config_id][feature] = anomalies

    answers = []

    for i, row in test_df.iterrows():
        config_id = row.configuration_item_id
        config_data = df[df.configuration_item_id == config_id]
        config_data.reset_index(drop=True, inplace=True)
        anomalies_ = list(zip(*all_anomalies[config_id].values()))
        id_in_anomalies = config_data[config_data.id == row.id].index[0]
        answers.append(list(anomalies_[id_in_anomalies]))

    result_df = pd.DataFrame({"id": test_df.id.values.tolist(), "target": answers})
    result_df.to_csv("case2.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection script.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=GENERAL["data_dir"],
        help="Directory containing train and test CSV files.",
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default=GENERAL["train_filename"],
        help="Filename of the training data CSV.",
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        default=GENERAL["test_filename"],
        help="Filename of the testing data CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.train_filename, args.test_filename)
