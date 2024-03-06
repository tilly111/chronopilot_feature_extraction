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

    :param tmp_data: Raw temperature data in a pandas data frame containing three columns: "LocalTimestamp", "TH"
                        and "T1".
    :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability, e.g.,
                        100 Hz.
    :param verbose: True gives debug prints.

    :return wd: working directories, temporary save object.
    :return m: features of the PPG time series.

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
    for k in raw_data_th.keys():
        if "time" in k.lower():
            raw_data_th = raw_data_th.rename(columns={k: "LocalTimestamp"})

    # merge data frames
    raw_data = pd.concat([raw_data_t1, raw_data_th], axis=0)
    raw_data.sort_values(by=['LocalTimestamp'], inplace=True)
    raw_data["T1"].interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    raw_data["TH"].interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    # down sample to target frequency -> frequency increased by x 2 because of merging and time steps are not aligned
    raw_data = raw_data.iloc[::2, :]

    return raw_data
