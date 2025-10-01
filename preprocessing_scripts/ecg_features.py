import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.signal import resample


def calculate_ecg_features_nk(ecg_data: pd.DataFrame, target_f=100, verbose=False):
    """ Calculates ecg_nk features using the neurokit library.

    Filters the raw ECG data and calculates features from it.

    :param ecg_data: Raw ECG data in a pandas data frame containing two columns: "LocalTimestamp" and "ECG".
    :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability, e.g.,
                        100 Hz.
    :param verbose: True gives debug prints.

    :return wd: working directories, temporary save object.
    :return m: features of the ECG time series.

    :raises Any Errors: ...
    """
    if verbose:
        print(ecg_data["LocalTimestamp"].iloc[-1], ecg_data["LocalTimestamp"].iloc[0],
              ecg_data["LocalTimestamp"].iloc[-1] - ecg_data["LocalTimestamp"].iloc[0])
    
    og_f = ecg_data["ECG"].to_numpy().shape[0] / (
        ecg_data["LocalTimestamp"].iloc[-1] - ecg_data["LocalTimestamp"].iloc[0])
    
    if verbose:
        print(f"original frequency {og_f}")
    
    signal_resampled = resample(ecg_data["ECG"].to_numpy(),
                                int(np.ceil((ecg_data["ECG"].to_numpy()).shape[0] * target_f / og_f)))
    f = signal_resampled.shape[0] / (ecg_data["LocalTimestamp"].iloc[-1] - ecg_data["LocalTimestamp"].iloc[0])
    
    if verbose:
        print(f"The original frequency is {og_f}")
        print(f"The new frequency is {f}")
        print(f"Length of the resampled signal {signal_resampled.shape}")
    
    p_1_process, info = nk.ecg_process(signal_resampled, sampling_rate=f)
    p_1_features = nk.ecg_analyze(p_1_process, sampling_rate=f, method="interval-related")
    
    return p_1_features


def transform_ecg(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the columns of a data frame such that they work with the rest of the pipeline.

    :param raw_data: Raw data frame which needs to be renamed.

    :return raw_data: Renamed data frame.

    :raises Any Errors: ...
    """

    # make "time stamp label" -> LocalTimestamp
    for k in raw_data.keys():
        if "time" in k.lower():
            raw_data = raw_data.rename(columns={k: "LocalTimestamp"})
            # make local timestamp start at 0
            raw_data["LocalTimestamp"] = raw_data["LocalTimestamp"] - raw_data["LocalTimestamp"].iloc[0]
        elif "ecg" in k.lower():
            raw_data = raw_data.rename(columns={k: "ECG"})

    return raw_data