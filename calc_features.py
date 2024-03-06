import os
import constants
import pandas as pd

from preprocessing_scripts.ppg_features import calculate_ppg_features, transform_ppg, calculate_ppg_features_nk
from preprocessing_scripts.eda_features import calculate_eda_features, transform_eda
from preprocessing_scripts.tmp_features import transform_thermo_pile, calculate_thermo_pile_features


########################################################################################################################
# Names
########################################################################################################################
block_name = "exp_S"  # "baseline", "practice", "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"
study = "2"  # "1" or "2"
neuro_kit = True


if study == "1":
    subjects = constants.SUBJECTS_STUDY_1
else:
    subjects = constants.SUBJECTS_STUDY_2

########################################################################################################################
# feature extraction -- PPG
########################################################################################################################
if neuro_kit:
    ppg_features = pd.DataFrame(columns=["subject"] + constants.ALL_PPG_FEATURES_NEUROKIT)
else:
    ppg_features = pd.DataFrame(columns=["subject"] + constants.ALL_PPG_FEATURES_HEARTPY)

if not os.path.exists(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/"):
    os.makedirs(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/")

for i, subject in enumerate(subjects):
    print(f"Subject: {subject}")
    df = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}/PG/subject-{subject}_PG.csv")
    df["time"] = df["time"] - df["time"][0]
    bg_block = (df.loc[(df["block"] == block_name)]).reset_index()
    freq = bg_block.shape[0] / (bg_block["time"].iloc[-1] - bg_block["time"].iloc[0])
    print(f"Frequency: {freq}")
    bg_block = transform_ppg(bg_block)
    if neuro_kit:
        m = calculate_ppg_features_nk(bg_block, verbose=False)

        ppg_features.loc[i] = [subject] + [m[k].loc[0] for k in m.keys()]
    else:
        wd, m = calculate_ppg_features(bg_block, target_f=freq, verbose=False)

        ppg_features.loc[i] = [subject] + [m[k] for k in m.keys()]
if neuro_kit:
    ppg_features.to_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/ppg_nk.csv", index=False)
else:
    ppg_features.to_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/ppg.csv", index=False)

########################################################################################################################
# feature extraction -- EDA
########################################################################################################################
eda_features = pd.DataFrame(columns=["subject"] + constants.ALL_EDA_FEATURES)

if not os.path.exists(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/"):
    os.makedirs(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/")

for i, subject in enumerate(subjects):
    print(f"Subject: {subject}")
    df = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}/EA/subject-{subject}_EA.csv")
    df["time"] = df["time"] - df["time"][0]
    bg_block = (df.loc[(df["block"] == block_name)]).reset_index()
    freq = bg_block.shape[0] / (bg_block["time"].iloc[-1] - bg_block["time"].iloc[0])
    print(f"Frequency: {freq}")
    bg_block = transform_eda(bg_block)
    m = calculate_eda_features(bg_block, verbose=False)

    eda_features.loc[i] = [subject] + [m[k].loc[0] for k in m.keys()]

eda_features.to_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/eda.csv", index=False)

########################################################################################################################
# feature extraction -- Thermo pile
########################################################################################################################
tmp_features = pd.DataFrame(columns=["subject"] + constants.ALL_TMP_FEATURES)

if not os.path.exists(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/"):
    os.makedirs(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/")

for i, subject in enumerate(subjects):
    print(f"Subject: {subject}")
    df_t1 = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}/T1/subject-{subject}_T1.csv")
    df_th = pd.read_csv(constants.SCREAM_DATA_PATH + f"study{study}/TH/subject-{subject}_TH.csv")
    df_t1["time"] = df_t1["time"] - df_t1["time"][0]
    df_th["time"] = df_th["time"] - df_th["time"][0]
    bg_block_t1 = (df_t1.loc[(df_t1["block"] == block_name)]).reset_index()
    bg_block_th = (df_th.loc[(df_th["block"] == block_name)]).reset_index()

    freq = bg_block_t1.shape[0] / (bg_block_t1["time"].iloc[-1] - bg_block_t1["time"].iloc[0])
    print(f"Frequency: {freq}")
    bg_block = transform_thermo_pile(bg_block_t1, bg_block_th)
    m = calculate_thermo_pile_features(bg_block, verbose=False)

    tmp_features.loc[i] = [subject] + [m[k].loc[0] for k in m.keys()]

tmp_features.to_csv(constants.SCREAM_DATA_PATH + f"study{study}_features/{block_name}/tmp.csv", index=False)
