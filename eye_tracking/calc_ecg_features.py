# /Volumes/Data/chronopilot/scream_experiment/study1_ts/exp_MA/subject-2_eda.csv

import platform
import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np

import constants
import pandas as pd

import heartpy as hp
import neurokit2 as nk

from preprocessing_scripts.ppg_features import calculate_ppg_features, transform_ppg, calculate_ppg_features_nk
from preprocessing_scripts.eda_features import calculate_eda_features, transform_eda
from preprocessing_scripts.tmp_features import transform_thermo_pile, calculate_thermo_pile_features


# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')

########################################################################################################################
# Names
########################################################################################################################
neuro_kit = True

JULIA_TIMES = [1, 3, 5]
JULIA_ROBOTS = [1, 3, 5, 7, 9, 11, 13, 15]
JULIA_PARTICIPANTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

if neuro_kit:
    # TODO adjust constant
    all_features_experiment = pd.DataFrame(columns=["time", "robot", "participant"] + constants.ALL_ECG_FEATURES_NEUROKIT)
    all_features_baseline = pd.DataFrame(columns=["time", "robot", "participant"] + constants.ALL_ECG_FEATURES_NEUROKIT)
    all_features_sub = pd.DataFrame(columns=["time", "robot", "participant"] + constants.ALL_ECG_FEATURES_NEUROKIT)
else:
    all_features_experiment = pd.DataFrame(columns=["time", "robot", "participant"] + constants.ALL_PPG_FEATURES_HEARTPY)
    all_features_baseline = pd.DataFrame(columns=["time", "robot", "participant"] + constants.ALL_PPG_FEATURES_HEARTPY)
    all_features_sub = pd.DataFrame(columns=["time", "robot", "participant"] + constants.ALL_PPG_FEATURES_HEARTPY)

for t in JULIA_TIMES:
    for r in JULIA_ROBOTS:
        for p in JULIA_PARTICIPANTS:
            print(f"Time: {t}, Robot: {r}, Participant: {p}")
            try:
                df_baseline = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/physio/{t}-{r}/{p}-{t}_{r}_ecg_baseline.csv")
                df_experiment = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/physio/{t}-{r}/{p}-{t}_{r}_ecg_experiment.csv")

                # plt.figure()
                # plt.plot(df_baseline["timestamp"], df_baseline["channel_0"], label="baseline")
                # plt.plot(df_experiment["timestamp"], df_experiment["channel_0"], label="experiment")
                # plt.legend()

                sampling_rate_exp = df_experiment.shape[0] / (t * 60)
                sampling_rate_bsl = df_baseline.shape[0] / (t * 60)

                if neuro_kit:
                    process_exp, info = nk.ecg_process(df_experiment["channel_0"].to_numpy(), sampling_rate=sampling_rate_exp)
                    features_exp = nk.ecg_analyze(process_exp, sampling_rate=sampling_rate_exp, method="interval-related")

                    process_bsl, info = nk.ecg_process(df_baseline["channel_0"].to_numpy(), sampling_rate=sampling_rate_bsl)
                    features_bsl = nk.ecg_analyze(process_bsl, sampling_rate=sampling_rate_bsl, method="interval-related")

                    all_features_experiment.loc[all_features_experiment.shape[0]] = [t, r, p] + [features_exp[k].loc[0] for k in features_exp.keys()]
                    all_features_baseline.loc[all_features_baseline.shape[0]] = [t, r, p] + [features_bsl[k].loc[0] for k in features_bsl.keys()]
                    all_features_sub.loc[all_features_sub.shape[0]] = [t, r, p] + [features_exp[k].loc[0] - features_bsl[k].loc[0] for k in features_exp.keys()]

                else:
                    wd_bsl, m_bsl = hp.process(df_baseline["channel_0"].to_numpy(), sample_rate=sampling_rate_bsl,
                                               high_precision=False, clean_rr=True, bpmmin=0, bpmmax=250)
                    wd_exp, m_exp = hp.process(df_experiment["channel_0"].to_numpy(), sample_rate=sampling_rate_exp,
                                       high_precision=False, clean_rr=True, bpmmin=0, bpmmax=250)

                    # for measure in m.keys():
                    #     print('%s: %f' % (measure, m[measure]))
                    # tmp_features_experiments = pd.DataFrame([t, r, p] + [m_exp[k] for k in m_exp.keys()])
                    # tmp_features_baseline = pd.DataFrame([t, r, p] + [m_bsl[k] for k in m_bsl.keys()])
                    # tmp_features_sub = pd.DataFrame([t, r, p] + [m_exp[k] - m_bsl[k] for k in m_exp.keys()])

                    all_features_experiment.loc[all_features_experiment.shape[0]] = [t, r, p] + [m_exp[k] for k in m_exp.keys()]
                    all_features_baseline.loc[all_features_baseline.shape[0]] = [t, r, p] + [m_bsl[k] for k in m_bsl.keys()]
                    all_features_sub.loc[all_features_sub.shape[0]] = [t, r, p] + [m_exp[k] - m_bsl[k] for k in m_exp.keys()]
                    # ppg_features.loc[time_slice] = [subject] + [m[k] for k in m.keys()]

                # plt.show()
            except:
                if neuro_kit:
                    pass
                else:
                    all_features_experiment.loc[all_features_experiment.shape[0]] = [t, r, p] + [np.NAN for _ in range(len(constants.ALL_PPG_FEATURES_HEARTPY))]
                    all_features_baseline.loc[all_features_baseline.shape[0]] = [t, r, p] + [np.NAN for _ in range(len(constants.ALL_PPG_FEATURES_HEARTPY))]
                    all_features_sub.loc[all_features_sub.shape[0]] = [t, r, p] + [np.NAN for _ in range(len(constants.ALL_PPG_FEATURES_HEARTPY))]
                print("setting does not exist")

if neuro_kit:
    all_features_experiment.to_csv("/Volumes/Data/chronopilot/Julia_study/features/ecg_features_experiment_nk.csv", index=False)
    all_features_baseline.to_csv("/Volumes/Data/chronopilot/Julia_study/features/ecg_features_baseline_nk.csv", index=False)
    all_features_sub.to_csv("/Volumes/Data/chronopilot/Julia_study/features/ecg_features_sub_nk.csv", index=False)
else:
    all_features_experiment.to_csv("/Volumes/Data/chronopilot/Julia_study/features/ecg_features_experiment.csv", index=False)
    all_features_baseline.to_csv("/Volumes/Data/chronopilot/Julia_study/features/ecg_features_baseline.csv", index=False)
    all_features_sub.to_csv("/Volumes/Data/chronopilot/Julia_study/features/ecg_features_sub.csv", index=False)
