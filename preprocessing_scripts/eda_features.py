import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample
import constants


def calculate_eda_features(eda_data: pd.DataFrame, target_f=10000, verbose=False):
    """ Calculates eda features using the NeuroKit2 library.

        Filters the raw EDA data and calculates features from it.

        :param eda_data: Raw EDA data in a pandas data frame containing two columns: "LocalTimestamp" and "EDA".
        :param target_f: Target frequency in Hz on how much to upsample the signal for better calculation stability,
                            e.g., 10000 Hz.
        :param verbose: True gives debug prints.

        :return p_1_features: EDA features of the raw time series.

        :raises Any Errors: ...
        """
    # catch if there are no EDA data
    if eda_data["EDA"].to_numpy().shape[0] < 2:
        warnings.warn("No EDA data -- returning nan")
        return pd.DataFrame(np.empty((len(constants.ALL_EDA_FEATURES),)).fill(np.nan), columns=constants.ALL_EDA_FEATURES, index=[0])

    og_f = eda_data["EDA"].to_numpy().shape[0] / (
            eda_data["LocalTimestamp"].iloc[-1] - eda_data["LocalTimestamp"].iloc[0])

    signal_resampled = resample(eda_data["EDA"].to_numpy(),
                                int(np.ceil((eda_data["EDA"].to_numpy()).shape[0] * target_f / og_f)))
    f = signal_resampled.shape[0] / (eda_data["LocalTimestamp"].iloc[-1] - eda_data["LocalTimestamp"].iloc[0])

    if verbose:
        print(f"The original frequency is {og_f}")
        print(f"The new frequency is {f}")
        print(f"Length of the resampled signal {signal_resampled.shape}")

    p_1_process, info = nk.eda_process(signal_resampled, sampling_rate=f)

    p_1_features = nk.eda_analyze(p_1_process, sampling_rate=f)

    return p_1_features


def transform_eda(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the columns of a data frame such that they work with the rest of the pipeline.

    :param raw_data: Raw data frame which needs to be renamed.

    :return raw_data: Renamed data frame.

    :raises Any Errors: ...
    """

    # make "time stamp label" -> LocalTimestamp
    for k in raw_data.keys():
        if "time" in k.lower():
            raw_data = raw_data.rename(columns={k: "LocalTimestamp"})
        elif "ea" in k.lower():
            raw_data = raw_data.rename(columns={k: "EDA"})

    return raw_data
