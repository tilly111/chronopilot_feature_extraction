import platform
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils.data_loader import Data_Loader
import constants

from preprocessing_scripts.ppg_features import transform_ppg, calculate_ppg_features, calculate_ppg_features_nk
from preprocessing_scripts.eda_features import transform_eda, calculate_eda_features
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
experiment_name = "helicopter_experiment"
interval = "start_posttest"


########################################################################################################################
# Load experiment data
########################################################################################################################
labels = pd.read_csv("/Volumes/Data/chronopilot/helicopter/new_label/labels_3_classes.csv", sep=";")
time_stamps = pd.read_csv("/Volumes/Data/chronopilot/helicopter/Physiological/timestamps.csv")

data_loader = Data_Loader("data/")

## obtain all time steps
pretests = time_stamps.loc[time_stamps["Phase"] == "pretest"]["LocalTimestamp"].to_numpy()
start = time_stamps.loc[time_stamps["Phase"] == "start"]["LocalTimestamp"].to_numpy()
takeoff = time_stamps.loc[time_stamps["Phase"] == "takeoff"]["LocalTimestamp"].to_numpy()
p1 = time_stamps.loc[time_stamps["Phase"] == "p1"]["LocalTimestamp"].to_numpy()
p1End = time_stamps.loc[time_stamps["Phase"] == "p1End"]["LocalTimestamp"].to_numpy()
p2 = time_stamps.loc[time_stamps["Phase"] == "p2"]["LocalTimestamp"].to_numpy()
p2End = time_stamps.loc[time_stamps["Phase"] == "p2End"]["LocalTimestamp"].to_numpy()
landing = time_stamps.loc[time_stamps["Phase"] == "landing"]["LocalTimestamp"].to_numpy()
posttest = time_stamps.loc[time_stamps["Phase"] == "posttest"]["LocalTimestamp"].to_numpy()

########################################################################################################################
# feature extraction -- PPG (neurokit)
########################################################################################################################
# check if directory exists for writing
if not os.path.exists("./preprocessed_data/" + experiment_name + "/ppg_nk/" + interval + "/"):
    os.makedirs("./preprocessed_data/" + experiment_name + "/ppg_nk/" + interval + "/")
data = data_loader.load_helicopter("PPG")
for i in range(12):  # for all participants
    # for saving
    neurokit_helicopter = pd.DataFrame(columns=constants.ALL_PPG_FEATURES_NEUROKIT_AVAILABLE)
    print(f"person {i+1}: pretest {takeoff[i*4:(i+1)*4] - start[i*4:(i+1)*4]}")

    for s in range(4):  # for all settings
        # cut data into pieces
        print(f"setting user {i+1} setting {s+1} ---------------------------")
        d = data[i][s][0]
        minuend = d.loc[((d["LocalTimestamp"] >= start[i*4 + s]) & (d["LocalTimestamp"] <= posttest[i*4 + s]))]
        subtrahend = d.loc[((d["LocalTimestamp"] >= start[i*4 + s]) & (d["LocalTimestamp"] <= takeoff[i*4 + s]))]

        # format data
        minuend = transform_ppg(minuend)
        subtrahend = transform_ppg(subtrahend)

        # calculate features
        if i == 10 and s == 1:  # hacking the only data point which is nan for some reason?
            min_m = calculate_ppg_features_nk(minuend, target_f=200, verbose=False)
            # sub_m = calculate_ppg_features_nk(subtrahend, target_f=200, verbose=False)
        else:
            min_m = calculate_ppg_features_nk(minuend, target_f=100, verbose=False)
            # sub_m = calculate_ppg_features_nk(subtrahend, target_f=100, verbose=False)

        # background subtraction
        # for k in min_m.keys():
        #     min_m[k] = min_m[k] - sub_m[k]
            # min_m[k] = sub_m[k] - min_m[k]
        # print(min_m.shape)
        # print(min_m.head())
        # print(neurokit_helicopter.shape)
        # print(neurokit_helicopter.head())

        # neurokit_helicopter.loc[s] = min_m.values
        # append the two data frames
        neurokit_helicopter = pd.concat([neurokit_helicopter, min_m], axis=0)
    # print results
    # print(f"participant: {i + 1} ---------------------------")
    # print(heartpy_helicopter)
    # or save the data
    neurokit_helicopter.to_csv("/Volumes/Data/chronopilot/helicopter/features/" + experiment_name + "/ppg_nk/" + interval + f"/subjectID_{i+1}.csv")
exit(111)
########################################################################################################################
# feature extraction -- PPG (HeartPy)
########################################################################################################################
# check if directory exists for writing
if not os.path.exists("./preprocessed_data/" + experiment_name + "/ppg_nk/" + interval + "/"):
    os.makedirs("./preprocessed_data/" + experiment_name + "/ppg_nk/" + interval + "/")
data = data_loader.load_helicopter("PPG")
for i in range(12):  # for all participants
    # for saving
    heartpy_helicopter = pd.DataFrame(columns=constants.ALL_PPG_FEATURES_HEARTPY)
    print(f"person {i+1}: pretest {takeoff[i*4:(i+1)*4] - start[i*4:(i+1)*4]}")

    for s in range(4):  # for all settings
        # cut data into pieces
        d = data[i][s][0]
        minuend = d.loc[((d["LocalTimestamp"] >= start[i*4 + s]) & (d["LocalTimestamp"] <= posttest[i*4 + s]))]
        subtrahend = d.loc[((d["LocalTimestamp"] >= start[i*4 + s]) & (d["LocalTimestamp"] <= takeoff[i*4 + s]))]

        # format data
        minuend = transform_ppg(minuend)
        subtrahend = transform_ppg(subtrahend)

        # calculate features
        if i == 10 and s == 1:  # hacking the only data point which is nan for some reason?
            min_wd, min_m = calculate_ppg_features(minuend, target_f=200, verbose=False)
            sub_wd, sub_m = calculate_ppg_features(subtrahend, target_f=200, verbose=False)
        else:
            min_wd, min_m = calculate_ppg_features(minuend, target_f=100, verbose=False)
            sub_wd, sub_m = calculate_ppg_features(subtrahend, target_f=100, verbose=False)

        # background subtraction
        for k in min_m.keys():
            min_m[k] = min_m[k] - sub_m[k]
            # min_m[k] = sub_m[k] - min_m[k]

        heartpy_helicopter.loc[s] = min_m
    # print results
    # print(f"participant: {i + 1} ---------------------------")
    # print(heartpy_helicopter)
    # or save the data
    heartpy_helicopter.to_csv(f"preprocessed_data/" + experiment_name + f"/ppg_nk/" + interval + f"/subjectID_{i+1}.csv")


########################################################################################################################
# feature extraction -- EDA
########################################################################################################################
if not os.path.exists("./preprocessed_data/" + experiment_name + "/eda/" + interval + "/"):
    os.makedirs("./preprocessed_data/" + experiment_name + "/eda/" + interval + "/")
data = data_loader.load_helicopter("EDA")
for i in range(12):  # for all participants
    # for saving
    eda_helicopter = pd.DataFrame(columns=constants.ALL_EDA_FEATURES)
    print(f"person {i+1}: pretest {takeoff[i*4:(i+1)*4] - start[i*4:(i+1)*4]}")

    for s in range(4):  # for all settings
        # cut data into pieces
        d = data[i][s][0]

        minuend = d.loc[
            ((d["LocalTimestamp"] >= start[i * 4 + s]) & (d["LocalTimestamp"] <= posttest[i * 4 + s]))]
        if i == 5 and s == 0:  # ACHTUNG Hack because the minimum required signal time is  10 sec and in this setting it is only 9
            subtrahend = d.loc[
                ((d["LocalTimestamp"] >= start[i * 4 + s] - 1.5) & (d["LocalTimestamp"] <= takeoff[i * 4 + s]))]
        else:
            subtrahend = d.loc[
                ((d["LocalTimestamp"] >= start[i * 4 + s]) & (d["LocalTimestamp"] <= takeoff[i * 4 + s]))]

        minuend = transform_eda(minuend)
        subtrahend = transform_eda(subtrahend)

        min_m = calculate_eda_features(minuend)
        sub_m = calculate_eda_features(subtrahend)

        # background subtraction
        for k in min_m.keys():
            min_m[k] = min_m[k] - sub_m[k]

        eda_helicopter.loc[s] = min_m.loc[0]
    # print results
    # print(f"participant: {i + 1} ---------------------------")
    # print(eda_helicopter)
    # or save the data
    eda_helicopter.to_csv(f"preprocessed_data/" + experiment_name + f"/eda/" + interval + f"/subjectID_{i+1}.csv")


########################################################################################################################
# feature extraction -- Thermo pile
########################################################################################################################
if not os.path.exists("./preprocessed_data/" + experiment_name + "/tmp/" + interval + "/"):
    os.makedirs("./preprocessed_data/" + experiment_name + "/tmp/" + interval + "/")
data_t1 = data_loader.load_helicopter("T1")
data_th = data_loader.load_helicopter("TH")
for i in range(12):  # for all participants
    # for saving
    tmp_helicopter = pd.DataFrame(columns=constants.ALL_TMP_FEATURES)
    print(f"person {i + 1}: pretest {takeoff[i * 4:(i + 1) * 4] - start[i * 4:(i + 1) * 4]}")

    for s in range(4):  # for all settings
        # cut data into pieces
        d_t1 = data_t1[i][s][0]
        minuend_t1 = d_t1.loc[
            ((d_t1["LocalTimestamp"] >= start[i * 4 + s]) & (d_t1["LocalTimestamp"] <= posttest[i * 4 + s]))]
        subtrahend_t1 = d_t1.loc[
            ((d_t1["LocalTimestamp"] >= start[i * 4 + s]) & (d_t1["LocalTimestamp"] <= takeoff[i * 4 + s]))]
        d_th = data_th[i][s][0]
        minuend_th = d_th.loc[
            ((d_th["LocalTimestamp"] >= start[i * 4 + s]) & (d_th["LocalTimestamp"] <= posttest[i * 4 + s]))]
        subtrahend_th = d_th.loc[
            ((d_th["LocalTimestamp"] >= start[i * 4 + s]) & (d_th["LocalTimestamp"] <= takeoff[i * 4 + s]))]

        minuend = transform_thermo_pile(minuend_t1, minuend_th)
        subtrahend = transform_thermo_pile(subtrahend_t1, subtrahend_th)

        # calcualte features
        min_m = calculate_thermo_pile_features(minuend)
        sub_m = calculate_thermo_pile_features(subtrahend)
        # background subtraction
        for k in min_m.keys():
            min_m[k] = min_m[k] - sub_m[k]

        tmp_helicopter.loc[s] = min_m.loc[0]
    # print results
    # print(f"participant: {i + 1} ---------------------------")
    # print(tmp_helicopter)
    # or save the data
    tmp_helicopter.to_csv(f"preprocessed_data/" + experiment_name + f"/tmp/" + interval + f"/subjectID_{i+1}.csv")