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
JULIA_TIMES = [1, 3, 5]
JULIA_ROBOTS = [1, 3, 5, 7, 9, 11, 13, 15]
JULIA_PARTICIPANTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

all_labels = pd.DataFrame(columns=["time", "robot", "participant", "duration_estimate", "ppot", "valence", "arousal", "flow", "task_difficulty"])

for p in JULIA_PARTICIPANTS:
    try:
        df = pd.read_csv(f"/Volumes/Data/chronopilot/Julia_study/Fragebogen_cleaned/{p}/all.csv", sep=";", header=None)
        for t in range(3):
            for r in range(8):
                all_labels.loc[all_labels.shape[0]] = [JULIA_TIMES[t], JULIA_ROBOTS[r], p] + df.iloc[(r)+8*t, 1:].to_list()
    except:
        for t in range(3):
            for r in range(8):
                all_labels.loc[all_labels.shape[0]] = [t, r, p] + [np.nan for _ in range(6)]
        print(f"Participant {p} not found")

all_labels.to_csv("/Volumes/Data/chronopilot/Julia_study/Fragebogen_cleaned/all_labels.csv", index=False)