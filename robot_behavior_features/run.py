import os
from pathlib import Path
import pandas as pd

import constants
from preprocessing_scripts.ecg_features import transform_ecg, calculate_ecg_features_nk
from preprocessing_scripts.eda_features import transform_eda, calculate_eda_features
from preprocessing_scripts.tmp_features import transform_tmp, calculate_tmp_features


def load_robot_behavior(kind="RR", path="/Volumes/Data/chronopilot/2022_robot_behavior"):
    labels = pd.read_csv(os.path.join(path, "Questionnaire_data.csv"), sep=';', header=0, decimal=',')
    data = [[[None for _ in range(3)] for _ in range(2)] for _ in range(25)]
    label_data = [[[None for _ in range(3)] for _ in range(2)] for _ in range(25)]
    
    for p in range(1, 26):  # for all subjects
        for s in range(1, 3):  # for all speeds
            for r in range(1, 4):  # for all robot numbers
                data_path = os.path.join(path, "Measurements_fixed", f"p_{p}", f"{s}_{r}", f"{kind}.csv")
                x = pd.read_csv(data_path, sep=';', header=0, decimal=',')
                ppot_id = str(s) + '.' + str(r) + '.2'  # perceived passage of time
                duration_estimate_id = str(s) + '.' + str(r) + '.1'  # actual duration
                y = pd.DataFrame(data={'ppot': labels[labels['Index'] == p][ppot_id],
                                       'duraton_estimate': labels[labels['Index'] == p][duration_estimate_id]})
                # print(y)
                data[p - 1][s - 1][r - 1] = x
                label_data[p - 1][s - 1][r - 1] = y
    return data, label_data


def estimate_label(value, target, n_classes):
    if target == "ppot":
        if n_classes == 3:
            if int(value["ppot"].iloc[0]) < 3:
                return 0
            elif int(value["ppot"].iloc[0]) == 3:
                return 1
            else:
                return 2
        elif n_classes == 2:
            if int(value["ppot"].iloc[0]) <= 3:
                return 0
            else:
                return 1
        else:
            raise ValueError("must be 2 or 3 classes for ppot")
    elif target == "duration_estimate":
        if n_classes == 3:
            if (float(value["duraton_estimate"]) - 180.0) / 180.0 <= 0.75:  # TODO check if this is correct
                return 0
            elif (float(value["duraton_estimate"]) - 180.0) / 180.0 > 0.75 and (
                float(value["duraton_estimate"]) - 180.0) / 180.0 <= 1.05:  # TODO check if this is correct
                return 1
            else:
                return 2
        elif n_classes == 2:
            if (float(value["duraton_estimate"]) - 180.0) / 180.0 <= 0.9:  # TODO check if this is correct
                return 0  # underestimated
            else:
                return 1  # overestimated
        else:
            raise ValueError("must be 2 or 3 classes for duration estimate")
    else:
        raise ValueError("Target not implemented")


def ecg_robot_behavior(dir_path, baseline_subtraction, upper_bound, target, n_classes) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_robot_behavior('ECG', path=dir_path)
    
    # for saving
    result_frame = pd.DataFrame(
        columns=["Participant", "Speed", "Robot", "Slice"] + constants.ALL_ECG_FEATURES_NEUROKIT)
    result_frame_label = pd.DataFrame(columns=["Participant", "Speed", "Robot", "Slice", f"{target}"])
    
    for p in range(25):  # for all subjects
        for s in range(2):  # for all speeds
            for r in range(3):  # for all robot numbers
                print(f"Participant {p + 1}, Speed {s + 1}, Robots {r + 1} ---------------------------")
                if x[p][s][r].shape[0] == 0:  # apperently we miss some data sometimes
                    print("No ECG data for this setting, continue...")
                    continue
                d_ecg = x[p][s][r]
                d_ecg["TimeStamp"] = d_ecg["TimeStamp"].clip(upper=upper_bound)
                
                # format data
                interval = transform_ecg(d_ecg)
                
                # calculate features
                interval_f = calculate_ecg_features_nk(interval, target_f=130, verbose=False)
                
                # baseline subtraction
                # TODO we do not have the data for that here...
                
                # append to the result frame
                data_row = {"Participant": p + 1, "Speed": s + 1, "Robot": r+1, "Slice": "all", **interval_f.iloc[0].to_dict()}
                result_frame.loc[len(result_frame)] = data_row
                
                # compute label
                l = estimate_label(labels[p][s][r], target, n_classes)
                result_frame_label.loc[len(result_frame_label)] = {"Participant": p + 1, "Speed": s + 1, "Robot": r+1,
                                                                   "Slice": "all",
                                                                   f"{target}": l}
    return result_frame, result_frame_label


def eda_robot_behavior(dir_path, baseline_subtraction, upper_bound, target, n_classes) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_robot_behavior('EDA', path=dir_path)
    
    # for saving
    result_frame = pd.DataFrame(
        columns=["Participant", "Speed", "Robot", "Slice"] + constants.ALL_TMP_FEATURES_WRIST)
    result_frame_label = pd.DataFrame(columns=["Participant", "Speed", "Robot", "Slice", f"{target}"])
    
    for p in range(25):  # for all subjects
        for s in range(2):  # for all speeds
            for r in range(3):  # for all robot numbers
                print(f"Participant {p + 1}, Speed {s + 1}, Robots {r + 1} ---------------------------")
                if x[p][s][r].shape[0] == 0:  # apperently we miss some data sometimes
                    print("No EDA data for this setting, continue...")
                    continue
                d_eda = x[p][s][r]
                d_eda["TimeStamp"] = d_eda["TimeStamp"].clip(upper=upper_bound)
                
                # format data
                interval = transform_eda(d_eda)
                
                # calculate features
                interval_f = calculate_eda_features(interval, target_f=4, verbose=False)
                
                # baseline subtraction
                # TODO we do not have the data for that here...
                
                # append to the result frame
                data_row = {"Participant": p + 1, "Speed": s + 1, "Robot": r + 1, "Slice": "all", **interval_f.iloc[0].to_dict()}
                result_frame.loc[len(result_frame)] = data_row
                
                # compute label
                l = estimate_label(labels[p][s][r], target, n_classes)
                result_frame_label.loc[len(result_frame_label)] = {"Participant": p + 1, "Speed": s + 1, "Robot": r+1,
                                                                   "Slice": "all",
                                                                   f"{target}": l}
    return result_frame, result_frame_label


def tmp_robot_behavior(dir_path, baseline_subtraction, upper_bound, target, n_classes) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_robot_behavior('TMP', path=dir_path)
    
    # for saving
    result_frame = pd.DataFrame(
        columns=["Participant", "Speed", "Robot", "Slice"] + constants.ALL_TMP_FEATURES)
    result_frame_label = pd.DataFrame(columns=["Participant", "Speed", "Robot", "Slice", f"{target}"])
    
    for p in range(25):  # for all subjects
        for s in range(2):  # for all speeds
            for r in range(3):  # for all robot numbers
                print(f"Participant {p + 1}, Speed {s + 1}, Robots {r + 1} ---------------------------")
                if x[p][s][r].shape[0] == 0:  # apperently we miss some data sometimes
                    print("No TMP data for this setting, continue...")
                    continue
                d_tmp = x[p][s][r]
                d_tmp["TimeStamp"] = d_tmp["TimeStamp"].clip(upper=upper_bound)
                
                # format data
                interval = transform_tmp(d_tmp)
                
                # calculate features
                interval_f = calculate_tmp_features(interval, target_f=4, verbose=False)
                
                # baseline subtraction
                # TODO we do not have the data for that here...
                
                # append to the result frame
                data_row = {"Participant": p + 1, "Speed": s + 1, "Robot": r + 1, "Slice": "all", **interval_f.iloc[0].to_dict()}
                result_frame.loc[len(result_frame)] = data_row
                
                # compute label
                l = estimate_label(labels[p][s][r], target, n_classes)
                result_frame_label.loc[len(result_frame_label)] = {"Participant": p + 1, "Speed": s + 1, "Robot": r+1,
                                                                   "Slice": "all",
                                                                   f"{target}": l}
    return result_frame, result_frame_label

def run(config):
    dir_path = config.dir_path
    baseline_subtraction = config.baseline_subtraction
    window_size = config.window_size
    n_classes = config.n_classes
    target = config.target
    upper_bound = 180  # 180 seconds  TODO make configurable?
    
    if baseline_subtraction:
        print("We have no baseline data for this study...")
        exit(0)
    
    # add experiment name to path
    dir_path = os.path.join(dir_path, "2022_robot_behavior")
    
    df_ecg, label_ecg = ecg_robot_behavior(dir_path, baseline_subtraction, upper_bound, target, n_classes)
    df_eda, label_eda = eda_robot_behavior(dir_path, baseline_subtraction, upper_bound, target, n_classes)
    df_tmp, label_tmp = tmp_robot_behavior(dir_path, baseline_subtraction, upper_bound, target, n_classes)
    
    print("ECG features shape:", df_ecg.shape)
    print("Labels shape:", label_ecg.shape)
    print("EDA features shape:", df_eda.shape)
    print("Labels shape:", label_eda.shape)
    print("TMP features shape:", df_tmp.shape)
    print("Labels shape:", label_tmp.shape)
    
    # merge all dataframes on participant, session, slice
    df = df_ecg.merge(df_eda, on=["Participant", "Speed", "Robot", "Slice"], how="outer")
    df = df.merge(df_tmp, on=["Participant", "Speed", "Robot", "Slice"], how="outer")
    
    df_label = label_ecg.merge(label_eda, on=["Participant", "Speed", "Robot", "Slice"], how="outer")
    df_label = df_label.merge(label_tmp, on=["Participant", "Speed", "Robot", "Slice"], how="outer")
    
    print(f"All features shape: {df.shape}")
    print(f"All labels shape: {df_label.shape}")
    
    # save to csv
    if baseline_subtraction:
        print("We have no baseline data for this study...")
    else:
        # check if directory exists
        save_path = Path(os.path.join(dir_path, "preprocessed_data/no_baseline_subtraction"))
        save_path.mkdir(parents=True, exist_ok=True)
        # save_path = os.path.join(dir_path, f"features/features_no_baseline_subtraction")
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        df.to_csv(os.path.join(save_path, f"X_{target}_{n_classes}_classes.csv"), index=False)
        df_label.to_csv(os.path.join(save_path, f"y_{target}_{n_classes}_classes.csv"), index=False)
