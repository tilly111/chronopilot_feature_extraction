import os

import matplotlib.pyplot as plt
import numpy as np
import constants
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.signal import resample
import neurokit2 as nk
from scipy.signal import welch


def calculate_thermo_pile_features(tmp_data: pd.DataFrame, target_f=7.5, verbose=False):
    """ Calculates thermo-pile features.
    ATTENTION: Needs to go with ALL_TMP_FEATURES

    :param tmp_data: Raw temperature data in a pandas data frame containing three columns: "LocalTimestamp", "TH"
                        and "T1".
    :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability, e.g.,
                        100 Hz.
    :param verbose: True gives debug prints.

    :return m: features of the TMP time series.

    :raises Any Errors: ...
    """
    # get data
    th = tmp_data["TH"].to_numpy()
    t1 = tmp_data["T1"].to_numpy()

    # calculate the difference and means
    mean_tmp_dif = np.mean(th - t1)
    mean_th = np.mean(th)
    mean_t1 = np.mean(t1)

    gradient = np.sum(np.gradient(th - t1))
    f, psd = welch(th - t1, fs=target_f, nperseg=50)  # get the psd
    psd_power = sum(psd)

    # make data frame
    temp = pd.DataFrame(
        {"mean_tmp_dif": [mean_tmp_dif], "mean_th": [mean_th], "mean_t1": [mean_t1], "gradient": [gradient],
         "psd_power": [psd_power]})

    return temp


def calculate_tmp_features(tmp_data: pd.DataFrame, target_f=7.5, verbose=False):
    """ Calculates temperature features from the wrist?!
    ATTENTION: Needs to go with ALL_TMP_FEATURES_WRIST

    :param tmp_data: Raw temperature data in a pandas data frame containing two columns: "LocalTimestamp" and "TMP"
    :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability, e.g.,
                        100 Hz.
    :param verbose: True gives debug prints.

    :return m: features of the TMP time series.

    :raises Any Errors: ...
    """
    # get data
    t = tmp_data["TMP"].to_numpy()
    
    if verbose:
        print(f"Length of the signal: {t.shape}")
        print(f"Original frequency: {t.shape[0] / (tmp_data['LocalTimestamp'].iloc[-1] - tmp_data['LocalTimestamp'].iloc[0])}")

    # calculate the difference and means
    mean_tmp = np.mean(t)
    mean_tmp_dif = np.mean(np.diff(t))
    std_tmp_dif = np.std(np.diff(t))
    
    gradient = np.sum(np.gradient(t))
    f, psd = welch(t, fs=target_f, nperseg=50)  # get the psd
    psd_power = sum(psd)

    # make data frame
    temp = pd.DataFrame(
        {"mean_tmp": [mean_tmp], "mean_tmp_dif": [mean_tmp_dif], "std_tmp_dif": [std_tmp_dif], "gradient": [gradient],
         "psd_power": [psd_power]})

    return temp


def transform_thermo_pile(raw_data_t1: pd.DataFrame, raw_data_th: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the columns of a data frame such that they work with the rest of the pipeline.

    :param raw_data: Raw data frame which needs to be renamed.

    :return raw_data: Renamed data frame.

    :raises Any Errors: ...
    """
    # catch wrong keys
    for k in raw_data_t1.keys():
        if "time" in k.lower():
            raw_data_t1 = raw_data_t1.rename(columns={k: "LocalTimestamp"})
            raw_data_t1["LocalTimestamp"] = raw_data_t1["LocalTimestamp"] - raw_data_t1["LocalTimestamp"].iloc[0]
    for k in raw_data_th.keys():
        if "time" in k.lower():
            raw_data_th = raw_data_th.rename(columns={k: "LocalTimestamp"})
            raw_data_th["LocalTimestamp"] = raw_data_th["LocalTimestamp"] - raw_data_th["LocalTimestamp"].iloc[0]

    # merge data frames
    raw_data = pd.concat([raw_data_t1, raw_data_th], axis=0)
    raw_data.sort_values(by=['LocalTimestamp'], inplace=True)
    raw_data["T1"].interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    raw_data["TH"].interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    # down sample to target frequency -> frequency increased by x 2 because of merging and time steps are not aligned
    raw_data = raw_data.iloc[::2, :]

    return raw_data


def transform_tmp(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the columns of a data frame such that they work with the rest of the pipeline.

    :param raw_data: Raw data frame which needs to be renamed.

    :return raw_data: Renamed data frame.

    :raises Any Errors: ...
    """

    # make "time stamp label" -> LocalTimestamp
    for k in raw_data.keys():
        if "time" in k.lower():
            raw_data = raw_data.rename(columns={k: "LocalTimestamp"})
            raw_data["LocalTimestamp"] = raw_data["LocalTimestamp"] - raw_data["LocalTimestamp"].iloc[0]
        elif "tmp" in k.lower():
            raw_data = raw_data.rename(columns={k: "TMP"})

    return raw_data