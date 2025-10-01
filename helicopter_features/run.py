import os
import numpy as np
import pandas as pd

import constants
from preprocessing_scripts.ppg_features import transform_ppg, calculate_ppg_features_nk
from preprocessing_scripts.eda_features import transform_eda, calculate_eda_features
from preprocessing_scripts.tmp_features import transform_thermo_pile, calculate_thermo_pile_features


def load_helicopter(kind="RR", path="/Volumes/Data/chronopilot/2024_helicopter") -> [list, list]:
    labels = pd.read_csv(os.path.join(path, "timings.csv"), header=0)
    data = [[None for _ in range(4)] for _ in range(12)]
    label_data = [[None for _ in range(4)] for _ in range(12)]
    
    for p in range(1, 13):  # for all subjects
        for s in range(1, 5):  # for all settings
            data_path = os.path.join(path, f"Physiological/" + kind + "/subject" + str(p) + "-" + str(
                s) + "_" + kind + ".csv")
            x = pd.read_csv(data_path, header=0)
            y = labels.loc[(labels['ParticipantID'] == p) & (labels['Session'] == s)]
            data[p - 1][s - 1] = x
            label_data[p - 1][s - 1] = y
    return data, label_data


def estimate_label(value, target, n_classes):
    if target == "ppot":
        if n_classes == 3:
            if int(value["PassageOfTimeSlowFast"].iloc[0]) < 3:
                return 0
            elif int(value["PassageOfTimeSlowFast"].iloc[0]) == 3:
                return 1
            else:
                return 2
        elif n_classes == 2:
            if int(value["PassageOfTimeSlowFast"].iloc[0]) <= 3:
                return 0
            else:
                return 1
        else:
            raise ValueError("must be 2 or 3 classes for ppot")
    elif target == "duration_estimate":
        if n_classes == 3:
            if (float(value["SessionTimeEstimated"]) - float(value["SessionTimeActual"])) / float(value[
                "SessionTimeActual"]) <= 0.75:  # TODO check if this is correct
                return 0
            elif (float(value["SessionTimeEstimated"]) - float(value["SessionTimeActual"])) / float(value["SessionTimeActual"]) > 0.75 and (
                float(value["SessionTimeEstimated"]) - float(value["SessionTimeActual"])) / float(value[
                "SessionTimeActual"]) <= 1.05:  # TODO check if this is correct
                return 1
            else:
                return 2
        elif n_classes == 2:
            if (float(value["SessionTimeEstimated"]) - float(value["SessionTimeActual"])) / float(value[
                "SessionTimeActual"]) <= 0.9:  # TODO check if this is correct
                return 0  # underestimated
            else:
                return 1  # overestimated
        else:
            raise ValueError("must be 2 or 3 classes for duration estimate")
    else:
        raise ValueError("Target not implemented")


def ppg_helicopter(dir_path, baseline_start, baseline_end, interval_start, interval_end,
                   baseline_subtraction, target, n_classes) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_helicopter("PPG", dir_path)
    
    # for saving
    result_frame = pd.DataFrame(
        columns=["Participant", "Session", "Slice"] + constants.ALL_PPG_FEATURES_NEUROKIT_AVAILABLE)
    result_frame_label = pd.DataFrame(columns=["Participant", "Session", "Slice", f"{target}"])
    
    for i in range(12):  # for all participants
        for s in range(4):  # for all settings
            # cut data into pieces
            print(f"setting user {i + 1} setting {s + 1} ---------------------------")
            d = x[i][s]
            interval = d.loc[
                ((d["LocalTimestamp"] >= interval_start[i * 4 + s]) & (d["LocalTimestamp"] <= interval_end[i * 4 + s]))]
            baseline = d.loc[
                ((d["LocalTimestamp"] >= baseline_start[i * 4 + s]) & (d["LocalTimestamp"] <= baseline_end[i * 4 + s]))]
            
            # format data
            interval = transform_ppg(interval)
            baseline = transform_ppg(baseline)
            
            # calculate features
            interval_f = calculate_ppg_features_nk(interval, target_f=25, verbose=False)
            baseline_f = calculate_ppg_features_nk(baseline, target_f=25, verbose=False)
            
            # background subtraction
            if baseline_subtraction:
                for k in interval_f.keys():
                    # ensure baseline is not nan
                    if not np.isnan(baseline_f[k]).any():
                        interval_f[k] = interval_f[k] - baseline_f[k]
            
            # append to the result frame
            data_row = {"Participant": i + 1, "Session": s + 1, "Slice": "all", **interval_f.iloc[0].to_dict()}
            result_frame.loc[len(result_frame)] = data_row
            
            # compute label
            l = estimate_label(labels[i][s], target, n_classes)
            result_frame_label.loc[len(result_frame_label)] = {"Participant": i + 1, "Session": s + 1, "Slice": "all",
                                                               f"{target}": l}
    
    return result_frame, result_frame_label


def eda_helicopter(dir_path, baseline_start, baseline_end, interval_start, interval_end,
                   baseline_subtraction, target, n_classes) -> [pd.DataFrame, pd.DataFrame]:
    x, labels = load_helicopter("EDA", dir_path)
    
    # for saving
    result_frame = pd.DataFrame(
        columns=["Participant", "Session", "Slice"] + constants.ALL_EDA_FEATURES)
    result_frame_label = pd.DataFrame(columns=["Participant", "Session", "Slice", f"{target}"])
    
    for i in range(12):  # for all participants
        for s in range(4):  # for all settings
            # cut data into pieces
            print(f"setting user {i + 1} setting {s + 1} ---------------------------")
            d = x[i][s]
            interval = d.loc[
                ((d["LocalTimestamp"] >= interval_start[i * 4 + s]) & (d["LocalTimestamp"] <= interval_end[i * 4 + s]))]
            baseline = d.loc[
                ((d["LocalTimestamp"] >= baseline_start[i * 4 + s]) & (d["LocalTimestamp"] <= baseline_end[i * 4 + s]))]
            
            # format data
            interval = transform_eda(interval)
            baseline = transform_eda(baseline)
            
            # calculate features
            interval_f = calculate_eda_features(interval, target_f=15, verbose=False)
            baseline_f = calculate_eda_features(baseline, target_f=15, verbose=False)
            
            # background subtraction
            if baseline_subtraction:
                for k in interval_f.keys():
                    # ensure baseline is not nan
                    if not np.isnan(baseline_f[k]).any():
                        interval_f[k] = interval_f[k] - baseline_f[k]
            
            # append to the return frame
            data_row = {"Participant": i + 1, "Session": s + 1, "Slice": "all", **interval_f.iloc[0].to_dict()}
            result_frame.loc[len(result_frame)] = data_row
            
            # compute label
            l = estimate_label(labels[i][s], target, n_classes)
            result_frame_label.loc[len(result_frame_label)] = {"Participant": i + 1, "Session": s + 1, "Slice": "all",
                                                               f"{target}": l}
    
    return result_frame, result_frame_label


def tmp_helicopter(dir_path, baseline_start, baseline_end, interval_start, interval_end,
                   baseline_subtraction, target, n_classes) -> [pd.DataFrame, pd.DataFrame]:
    x_t1, labels = load_helicopter("T1", dir_path)
    x_th, _ = load_helicopter("TH", dir_path)
    
    # for saving
    result_frame = pd.DataFrame(
        columns=["Participant", "Session", "Slice"] + constants.ALL_TMP_FEATURES)
    result_frame_label = pd.DataFrame(columns=["Participant", "Session", "Slice", f"{target}"])
    
    for i in range(12):  # for all participants
        for s in range(4):  # for all settings
            # cut data into pieces
            print(f"setting user {i + 1} setting {s + 1} ---------------------------")
            d_t1 = x_t1[i][s]
            d_th = x_th[i][s]
            interval_t1 = d_t1.loc[
                ((d_t1["LocalTimestamp"] >= interval_start[i * 4 + s]) & (
                    d_t1["LocalTimestamp"] <= interval_end[i * 4 + s]))]
            baseline_t1 = d_t1.loc[
                ((d_t1["LocalTimestamp"] >= baseline_start[i * 4 + s]) & (
                    d_t1["LocalTimestamp"] <= baseline_end[i * 4 + s]))]
            interval_th = d_th.loc[
                ((d_th["LocalTimestamp"] >= interval_start[i * 4 + s]) & (
                    d_th["LocalTimestamp"] <= interval_end[i * 4 + s]))]
            baseline_th = d_th.loc[
                ((d_th["LocalTimestamp"] >= baseline_start[i * 4 + s]) & (
                    d_th["LocalTimestamp"] <= baseline_end[i * 4 + s]))]
            
            # format data
            interval = transform_thermo_pile(interval_t1, interval_th)
            baseline = transform_thermo_pile(baseline_t1, baseline_th)
            
            # calculate features
            interval_f = calculate_thermo_pile_features(interval, target_f=7.5, verbose=False)
            baseline_f = calculate_thermo_pile_features(baseline, target_f=7.5, verbose=False)
            
            # background subtraction
            if baseline_subtraction:
                for k in interval_f.keys():
                    # ensure baseline is not nan
                    if not np.isnan(baseline_f[k]).any():
                        interval_f[k] = interval_f[k] - baseline_f[k]
            
            # append to the return frame
            data_row = {"Participant": i + 1, "Session": s + 1, "Slice": "all", **interval_f.iloc[0].to_dict()}
            result_frame.loc[len(result_frame)] = data_row
            
            # compute label
            l = estimate_label(labels[i][s], target, n_classes)
            result_frame_label.loc[len(result_frame_label)] = {"Participant": i + 1, "Session": s + 1, "Slice": "all",
                                                                f"{target}": l}
            
    return result_frame, result_frame_label


def run(config):
    dir_path = config.dir_path
    baseline_subtraction = config.baseline_subtraction
    window_size = config.window_size
    n_classes = config.n_classes
    target = config.target
    
    # add experiment name to path
    dir_path = os.path.join(dir_path, "2024_helicopter")
    
    # get time stamps of the experiment
    time_stamps = pd.read_csv(os.path.join(dir_path, "Physiological/timestamps.csv"))
    
    ## obtain all time steps
    # baseline: start - take-off
    # interval: start - posttest
    # pretests = time_stamps.loc[time_stamps["Phase"] == "pretest"]["LocalTimestamp"].to_numpy()
    start = time_stamps.loc[time_stamps["Phase"] == "start"]["LocalTimestamp"].to_numpy()
    takeoff = time_stamps.loc[time_stamps["Phase"] == "takeoff"]["LocalTimestamp"].to_numpy()
    # p1 = time_stamps.loc[time_stamps["Phase"] == "p1"]["LocalTimestamp"].to_numpy()
    # p1End = time_stamps.loc[time_stamps["Phase"] == "p1End"]["LocalTimestamp"].to_numpy()
    # p2 = time_stamps.loc[time_stamps["Phase"] == "p2"]["LocalTimestamp"].to_numpy()
    # p2End = time_stamps.loc[time_stamps["Phase"] == "p2End"]["LocalTimestamp"].to_numpy()
    # landing = time_stamps.loc[time_stamps["Phase"] == "landing"]["LocalTimestamp"].to_numpy()
    posttest = time_stamps.loc[time_stamps["Phase"] == "posttest"]["LocalTimestamp"].to_numpy()
    
    # calculcate the features
    df_ppg, df_label = ppg_helicopter(dir_path, start, takeoff, start, posttest, baseline_subtraction, target, n_classes)
    df_eda, _ = eda_helicopter(dir_path, start, takeoff, start, posttest, baseline_subtraction, target, n_classes)
    df_tmp, _ = tmp_helicopter(dir_path, start, takeoff, start, posttest, baseline_subtraction, target, n_classes)
    
    # merge all dataframes on participant, session, slice
    df = df_ppg.merge(df_eda, on=["Participant", "Session", "Slice"], how="outer")
    df = df.merge(df_tmp, on=["Participant", "Session", "Slice"], how="outer")
    
    print(f"All features shape: {df.shape}")
    print(f"All labels shape: {df_label.shape}")
    
    # save to csv
    if baseline_subtraction:
        # check if directory exists
        save_path = os.path.join(dir_path, f"features/features_baseline_subtraction")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(os.path.join(save_path, f"X_{target}_{n_classes}classes.csv"), index=False)
        df_label.to_csv(os.path.join(save_path, f"y_{target}_{n_classes}classes.csv"), index=False)
    else:
        # check if directory exists
        save_path = os.path.join(dir_path, f"features/features_no_baseline_subtraction")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(os.path.join(save_path, f"X_{target}_{n_classes}classes.csv"), index=False)
        df_label.to_csv(os.path.join(save_path, f"y_{target}_{n_classes}classes.csv"), index=False)
