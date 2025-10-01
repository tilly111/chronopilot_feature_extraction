import os
import numpy as np
import pandas as pd

import constants
from preprocessing_scripts.ppg_features import transform_ppg, calculate_ppg_features_nk
from preprocessing_scripts.eda_features import transform_eda, calculate_eda_features
from preprocessing_scripts.tmp_features import transform_thermo_pile, calculate_thermo_pile_features


def load_scream(kind="RR", study=2, path="/Volumes/Data/chronopilot/2024_scream/study2"):
    # NOTE needed for "replace"
    pd.set_option('future.no_silent_downcasting', True)
    
    # load labels
    labels = pd.read_csv(os.path.join(path, "Block_labels.csv"), header=0)
    
    if study == 2:
        subjects = constants.SUBJECTS_STUDY_2
        blocks = ["baseline", "practice", "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
    elif study == 1:
        subjects = constants.SUBJECTS_STUDY_1
        blocks = ["baseline", "practice", "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
    else:
        raise ValueError("study not recognized")
    
    # blocks = ["baseline", "practice", "exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]
    
    if kind == "PPG":
        data_path = os.path.join(path, "PG")
        column_name = "PG"
    elif kind == "EDA":
        data_path = os.path.join(path, "EA")
        column_name = "EA"
    elif kind == "T1" or kind == "TH":
        data_path = os.path.join(path, kind)
        column_name = kind
    else:
        raise ValueError("kind not recognized")
    
    data = [[None for _ in range(len(blocks))] for _ in range(len(subjects))]
    label_data = [[None for _ in range(len(blocks))] for _ in range(len(subjects))]
    
    for i, subject in enumerate(subjects):
        for j, block in enumerate(blocks):
            df = pd.read_csv(os.path.join(data_path, f"subject-{subject}_{column_name}.csv"))
            df["time"] = df["time"] - df["time"][0]
            block_df = (df.loc[(df["block"] == block)]).reset_index()
            block_df.drop(columns=["index", "subject_nr", "block"], inplace=True)
            data[i][j] = block_df
            
            y = labels.loc[(labels['subject'] == subject) & (labels['block'] == block)]
            y.drop(columns=["subject", "block", "Unnamed: 3", "PSE"], inplace=True)
            y.replace({"overestimation": 1, "underestimation": 0}, inplace=True)
            # check for empty dataframe
            if y.empty:
                y = np.nan
            else:
                y = y['block_estimation'].values[0]
            label_data[i][j] = y
            
    return data, label_data
    

def ppg_scream(dir_path, study, baseline_subtraction, target) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_scream("PPG", study, dir_path)
    
    result_frame = pd.DataFrame(columns=["Participant", "Block", "Slice"] + constants.ALL_PPG_FEATURES_NEUROKIT_AVAILABLE)
    result_frame_label = pd.DataFrame(columns=["Participant", "Block", "Slice", f"{target}"])
    
    if study == 2:
        subjects = constants.SUBJECTS_STUDY_2
    elif study == 1:
        subjects = constants.SUBJECTS_STUDY_1
    else:
        raise ValueError("study not recognized")
    
    for i, subject in enumerate(subjects):
        for j, block in enumerate(["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]):
            print(f"Running subject {subject}, block {block} ---------------------------")
            d = x[i][j+1] # +1 to skip baseline
            
            # format data
            interval = transform_ppg(d)
            
            # calculate features
            interval_f = calculate_ppg_features_nk(interval, target_f=25, verbose=False)
            
            # background subtraction
            if baseline_subtraction:
                baseline = transform_ppg(x[i][0])
                baseline_f = calculate_ppg_features_nk(baseline, target_f=25, verbose=False)
                for k in interval_f.keys():
                    # ensure baseline is not nan
                    if not np.isnan(baseline_f[k]).any():
                        interval_f[k] = interval_f[k] - baseline_f[k]
            
            # append to the result frame
            data_row = {"Participant": subject, "Block": block, "Slice": "all", **interval_f.iloc[0].to_dict()}
            result_frame.loc[len(result_frame)] = data_row
            
            # compute label
            result_frame_label.loc[len(result_frame_label)] = {"Participant": subject, "Block": block, "Slice": "all",
                                                               f"{target}": labels[i][j+1]}  # NOTE we already have the label
    
    return result_frame, result_frame_label


def eda_scream(dir_path, study, baseline_subtraction, target) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_scream("EDA", study, dir_path)
    
    result_frame = pd.DataFrame(
        columns=["Participant", "Block", "Slice"] + constants.ALL_EDA_FEATURES)
    result_frame_label = pd.DataFrame(columns=["Participant", "Block", "Slice", f"{target}"])
    
    if study == 2:
        subjects = constants.SUBJECTS_STUDY_2
    elif study == 1:
        subjects = constants.SUBJECTS_STUDY_1
    else:
        raise ValueError("study not recognized")
    
    for i, subject in enumerate(subjects):
        for j, block in enumerate(["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]):
            print(f"Running subject {subject}, block {block} ---------------------------")
            d = x[i][j + 1]  # +1 to skip baseline
            
            # format data
            interval = transform_eda(d)
            
            # calculate features
            interval_f = calculate_eda_features(interval, target_f=15, verbose=False)
            
            # background subtraction
            if baseline_subtraction:
                baseline = transform_eda(x[i][0])
                baseline_f = calculate_eda_features(baseline, target_f=15, verbose=False)
                for k in interval_f.keys():
                    # ensure baseline is not nan
                    if not np.isnan(baseline_f[k]).any():
                        interval_f[k] = interval_f[k] - baseline_f[k]
            
            # append to the result frame
            data_row = {"Participant": subject, "Block": block, "Slice": "all", **interval_f.iloc[0].to_dict()}
            result_frame.loc[len(result_frame)] = data_row
            
            # compute label
            result_frame_label.loc[len(result_frame_label)] = {"Participant": subject, "Block": block, "Slice": "all",
                                                               f"{target}": labels[i][
                                                                   j + 1]}  # NOTE we already have the label
    
    return result_frame, result_frame_label


def tmp_scream(dir_path, study, baseline_subtraction, target) -> [pd.DataFrame, pd.DataFrame]:
    x_t1, labels = load_scream("T1", study, dir_path)
    x_th, _ = load_scream("TH", study, dir_path)
    
    result_frame = pd.DataFrame(
        columns=["Participant", "Block", "Slice"] + constants.ALL_TMP_FEATURES)
    result_frame_label = pd.DataFrame(columns=["Participant", "Block", "Slice", f"{target}"])
    
    if study == 2:
        subjects = constants.SUBJECTS_STUDY_2
    elif study == 1:
        subjects = constants.SUBJECTS_STUDY_1
    else:
        raise ValueError("study not recognized")
    
    for i, subject in enumerate(subjects):
        for j, block in enumerate(["exp_T", "exp_MA", "exp_TU", "exp_PU", "exp_S"]):
            d_t1 = x_t1[i][j+1] # +1 to skip baseline
            d_th = x_th[i][j+1]

            # format data
            interval = transform_thermo_pile(d_t1, d_th)
            
            # calculate features
            interval_f = calculate_thermo_pile_features(interval, verbose=False)
            
            # background subtraction
            if baseline_subtraction:
                baseline = transform_thermo_pile(x_t1[i][0], x_th[i][0])
                baseline_f = calculate_thermo_pile_features(baseline, verbose=False)
                for k in interval_f.keys():
                    # ensure baseline is not nan
                    if not np.isnan(baseline_f[k]).any():
                        interval_f[k] = interval_f[k] - baseline_f[k]
                
            # append to the result frame
            data_row = {"Participant": subject, "Block": block, "Slice": "all", **interval_f.iloc[0].to_dict()}
            result_frame.loc[len(result_frame)] = data_row
            
            # compute label
            result_frame_label.loc[len(result_frame_label)] = {"Participant": subject, "Block": block,
                                                               "Slice": "all",
                                                               f"{target}": labels[i][
                                                                   j + 1]}  # NOTE we already have the label
    return result_frame, result_frame_label

def run(config):
    dir_path = config.dir_path
    baseline_subtraction = config.baseline_subtraction
    window_size = config.window_size
    n_classes = config.n_classes
    target = config.target
    
    if target not in ["duration_estimate"]:
        raise ValueError("We only have duration_estimate as target")
    if n_classes != 2:
        raise ValueError("We only support 2 classes")
    
    # add experiment name to path
    dir_path = os.path.join(dir_path, "2024_scream")
    
    # apperently there are 2 studies (1 & 2)
    for study in [1, 2]:
        if study == 1 and baseline_subtraction:
            print(f"We dont have baseline subtraction for study 1, assuming no baseline subtraction")
            baseline_subtraction_s = False
        else:
            baseline_subtraction_s = baseline_subtraction
            
        dir_path_s = os.path.join(dir_path, f"study{study}")
        
        df_ppg, df_label = ppg_scream(dir_path_s, study, baseline_subtraction_s, target)
        df_eda, _ = eda_scream(dir_path_s, study, baseline_subtraction_s, target)
        df_tmp, _ = tmp_scream(dir_path_s, study, baseline_subtraction_s, target)
        
        # merge all dataframes on participant, block, slice
        df = df_ppg.merge(df_eda, on=["Participant", "Block", "Slice"], how="outer")
        df = df.merge(df_tmp, on=["Participant", "Block", "Slice"], how="outer")
        
        print(f"All features shape: {df.shape}")
        print(f"All labels shape: {df_label.shape}")
        
        # save to csv
        if baseline_subtraction_s:
            # check if directory exists
            save_path = os.path.join(dir_path_s, f"features/features_baseline_subtraction")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_csv(os.path.join(save_path, f"X_{target}_{n_classes}classes.csv"), index=False)
            df_label.to_csv(os.path.join(save_path, f"y_{target}_{n_classes}classes.csv"), index=False)
        else:
            # check if directory exists
            save_path = os.path.join(dir_path_s, f"features/features_no_baseline_subtraction")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_csv(os.path.join(save_path, f"X_{target}_{n_classes}classes.csv"), index=False)
            df_label.to_csv(os.path.join(save_path, f"y_{target}_{n_classes}classes.csv"), index=False)
    
        