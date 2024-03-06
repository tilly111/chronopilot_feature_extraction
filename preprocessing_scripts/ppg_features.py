import numpy as np
import pandas as pd
import heartpy as hp
import neurokit2 as nk
from scipy.signal import resample


def calculate_ppg_features(ppg_data: pd.DataFrame, target_f=100, verbose=False):
    """ Calculates ppg features using the heartPy library.

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

    wd, m = hp.process(signal_resampled, sample_rate=f, high_precision=True, clean_rr=True, bpmmin=0, bpmmax=250)

    # wd = working data, storing temporary values, m = measures aka feature
    return wd, m


def calculate_ppg_features_nk(ppg_data: pd.DataFrame, target_f=100, verbose=False):
    """ Calculates ppg features using the heartPy library.

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

    p_1_process, info = nk.ppg_process(signal_resampled, sampling_rate=f)

    p_1_features = nk.ppg_analyze(p_1_process, sampling_rate=f)

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
