import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heartpy as hp
import neurokit2 as nk
from scipy.signal import resample


def calculate_ppg_features(ppg_data: pd.DataFrame, target_f=100, verbose=False):
    """ Calculates ppg_nk features using the heartPy library.

    Filters the raw PPG data and calculates features from it.

    :param ppg_data: Raw PPG data in a pandas data frame containing two columns: "LocalTimestamp" and "PG".
    :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability, e.g.,
                        100 Hz.
    :param verbose: True gives debug prints.

    :return wd: working directories, temporary save object.
    :return m: features of the PPG time series.

    :raises Any Errors: ...
    """
    if verbose:
        print(ppg_data["LocalTimestamp"].iloc[-1], ppg_data["LocalTimestamp"].iloc[0],
              ppg_data["LocalTimestamp"].iloc[-1] - ppg_data["LocalTimestamp"].iloc[0])

    og_f = ppg_data["PG"].to_numpy().shape[0] / (
                ppg_data["LocalTimestamp"].iloc[-1] - ppg_data["LocalTimestamp"].iloc[0])

    if verbose:
        print(f"original frequency {og_f}")

    # how to choose the cutoff frequencies: below 0.7Hz (42 BPM) and above 3.5 Hz (210 BPM)
    filtered = hp.filter_signal(ppg_data["PG"].to_numpy(), [0.7, 3.5], sample_rate=og_f, order=3, filtertype='bandpass')

    signal_resampled = resample(filtered, int(np.ceil(filtered.shape[0] * target_f / og_f)))

    f = signal_resampled.shape[0] / (ppg_data["LocalTimestamp"].iloc[-1] - ppg_data["LocalTimestamp"].iloc[0])
    if verbose:
        print(f"actual target frequency: {f}")
        print(f"resampled shape {signal_resampled.shape}")
        print(f"original shape {ppg_data.shape}")
        print(f"original {ppg_data.head()}")
        # plt.figure()
        # # plt.plot(ppg_data["PG"], label="original")
        # plt.plot(filtered, label="filtered")
        # plt.plot(signal_resampled, label="resampled")
        # plt.legend()
        # plt.show()

    wd, m = hp.process(signal_resampled, sample_rate=f, high_precision=True, clean_rr=True, bpmmin=0, bpmmax=250)

    if verbose:
        hp.plotter(wd, m, show=True)
        plt.show()

    # wd = working data, storing temporary values, m = measures aka feature
    return wd, m


def calculate_ppg_features_nk(ppg_data: pd.DataFrame, target_f=100, verbose=False):
    """ Calculates ppg_nk features using the neurokit library.

    Filters the raw PPG data and calculates features from it.

    :param ppg_data: Raw PPG data in a pandas data frame containing two columns: "LocalTimestamp" and "PG".
    :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability, e.g.,
                        100 Hz.
    :param verbose: True gives debug prints.

    :return wd: working directories, temporary save object.
    :return m: features of the PPG time series.

    :raises Any Errors: ...
    """
    if verbose:
        print(ppg_data["LocalTimestamp"].iloc[-1], ppg_data["LocalTimestamp"].iloc[0],
              ppg_data["LocalTimestamp"].iloc[-1] - ppg_data["LocalTimestamp"].iloc[0])

    og_f = ppg_data["PG"].to_numpy().shape[0] / (
                ppg_data["LocalTimestamp"].iloc[-1] - ppg_data["LocalTimestamp"].iloc[0])

    if verbose:
        print(f"original frequency {og_f}")

    signal_resampled = resample(ppg_data["PG"].to_numpy(),
                                int(np.ceil((ppg_data["PG"].to_numpy()).shape[0] * target_f / og_f)))
    f = signal_resampled.shape[0] / (ppg_data["LocalTimestamp"].iloc[-1] - ppg_data["LocalTimestamp"].iloc[0])

    if verbose:
        print(f"The original frequency is {og_f}")
        print(f"The new frequency is {f}")
        print(f"Length of the resampled signal {signal_resampled.shape}")

    p_1_process, info = nk.ppg_process(signal_resampled, sampling_rate=f)  # , report=f"ppg_report_{int(target_f)}.html" -> somewhat broken
    p_1_features = nk.ppg_analyze(p_1_process, sampling_rate=f, method="interval-related")

    if verbose:
        nk.ppg_plot(p_1_process, info)
        plt.show()

    if og_f < 300:
        # we cannot calculate certain features due to the fact that the signal is too short or the sampling frequency is to small
        p_1_features.drop(columns=["HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5", "HRV_ULF"], inplace=True)

        try:
            p_1_features.drop(columns=['HRV_DFA_alpha2', 'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak',
                             'HRV_MFDFA_alpha2_Mean', 'HRV_MFDFA_alpha2_Max',
                             'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
                             'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment',], inplace=True)
        except:
            print("subject does not can calculate HRV_DFA_alpha2 values")

    return p_1_features


def transform_ppg(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the columns of a data frame such that they work with the rest of the pipeline.

    :param raw_data: Raw data frame which needs to be renamed.

    :return raw_data: Renamed data frame.

    :raises Any Errors: ...
    """

    # make "time stamp label" -> LocalTimestamp
    for k in raw_data.keys():
        if "time" in k.lower():
            raw_data = raw_data.rename(columns={k: "LocalTimestamp"})
        elif "pg" in k.lower():
            raw_data = raw_data.rename(columns={k: "PG"})

    return raw_data
