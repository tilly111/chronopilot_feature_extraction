import platform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing_scripts.eye_features import calc_pupil_features_tw, calc_fixation_features_tw

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

confidence_threshold = 0.9

all_exp_fixations_features = pd.DataFrame(
    columns=["time", "robot", "participant", "slice"] + ["fixation_frequency", "fixation_duration_mean",
                                                         "fixation_duration_max", "fixation_dispersion_mean",
                                                         "fixation_dispersion_max", "saccade_frequency",
                                                         "saccade_durations_mean", "saccade_durations_max",
                                                         "saccade_speed_mean", "saccade_speed_max"])
all_exp_pupil_features = pd.DataFrame(columns=["time", "robot", "participant", "slice"] + ["pupil_diameter0_2d_mean",
                                                                                           "pupil_diameter0_2d_max",
                                                                                           "pupil_diameter0_2d_dev",
                                                                                           "pupil_diameter0_2d_ipa",
                                                                                           "pupil_diameter1_2d_mean",
                                                                                           "pupil_diameter1_2d_max",
                                                                                           "pupil_diameter1_2d_dev",
                                                                                           "pupil_diameter1_2d_ipa",
                                                                                           "pupil_diameter0_3d_mean",
                                                                                           "pupil_diameter0_3d_max",
                                                                                           "pupil_diameter0_3d_dev",
                                                                                           "pupil_diameter0_3d_ipa",
                                                                                           "pupil_diameter1_3d_mean",
                                                                                           "pupil_diameter1_3d_max",
                                                                                           "pupil_diameter1_3d_dev",
                                                                                           "pupil_diameter1_3d_ipa"])
tw = 20  # time window in seconds

for t in JULIA_TIMES:
    for r in JULIA_ROBOTS:
        for p in JULIA_PARTICIPANTS:
            print(f"Time: {t}, Robot: {r}, Participant: {p}")
            try:
                df_pupillometry_experiment = pd.read_csv(
                    f"/Volumes/Data/chronopilot/Julia_study/pupil/{t}-{r}/{p}-{t}_{r}_pupillometry_experiment.csv")
                df_pupillometry_baseline = pd.read_csv(
                    f"/Volumes/Data/chronopilot/Julia_study/pupil/{t}-{r}/{p}-{t}_{r}_pupillometry_baseline.csv")
                df_fixations_experiment = pd.read_csv(
                    f"/Volumes/Data/chronopilot/Julia_study/pupil/{t}-{r}/{p}-{t}_{r}_fixations_experiment.csv")
                df_fixations_baseline = pd.read_csv(
                    f"/Volumes/Data/chronopilot/Julia_study/pupil/{t}-{r}/{p}-{t}_{r}_fixations_baseline.csv")

                # remove not confident data
                df_pupillometry_baseline = df_pupillometry_baseline.loc[
                    df_pupillometry_baseline["confidence"] > confidence_threshold]
                df_pupillometry_experiment = df_pupillometry_experiment.loc[
                    df_pupillometry_experiment["confidence"] > confidence_threshold]
                df_fixations_baseline = df_fixations_baseline.loc[
                    df_fixations_baseline["confidence"] > confidence_threshold]
                df_fixations_experiment = df_fixations_experiment.loc[
                    df_fixations_experiment["confidence"] > confidence_threshold]

                # remove unlogical norm values
                df_pupillometry_baseline = df_pupillometry_baseline.loc[
                    (df_pupillometry_baseline["norm_pos_x"] >= 0) & (df_pupillometry_baseline["norm_pos_x"] <= 1.1)]
                df_pupillometry_experiment = df_pupillometry_experiment.loc[
                    (df_pupillometry_experiment["norm_pos_y"] >= 0) & (df_pupillometry_experiment["norm_pos_y"] <= 1.1)]
                df_fixations_baseline = df_fixations_baseline.loc[
                    (df_fixations_baseline["norm_pos_x"] >= 0) & (df_fixations_baseline["norm_pos_x"] <= 1.1)]
                df_fixations_experiment = df_fixations_experiment.loc[
                    (df_fixations_experiment["norm_pos_y"] >= 0) & (df_fixations_experiment["norm_pos_y"] <= 1.1)]

                # remove unlogical diameter values
                df_pupillometry_baseline.loc[df_pupillometry_baseline['diameter0_2d'] > 150, 'diameter0_2d'] = np.NAN
                df_pupillometry_baseline.loc[df_pupillometry_baseline['diameter1_2d'] > 150, 'diameter1_2d'] = np.NAN
                df_pupillometry_baseline.loc[df_pupillometry_baseline['diameter0_3d'] > 10, 'diameter0_3d'] = np.NAN
                df_pupillometry_baseline.loc[df_pupillometry_baseline['diameter1_3d'] > 10, 'diameter1_3d'] = np.NAN

                df_pupillometry_experiment.loc[
                    df_pupillometry_experiment['diameter0_2d'] > 150, 'diameter0_2d'] = np.NAN
                df_pupillometry_experiment.loc[
                    df_pupillometry_experiment['diameter1_2d'] > 150, 'diameter1_2d'] = np.NAN
                df_pupillometry_experiment.loc[df_pupillometry_experiment['diameter0_3d'] > 10, 'diameter0_3d'] = np.NAN
                df_pupillometry_experiment.loc[df_pupillometry_experiment['diameter1_3d'] > 10, 'diameter1_3d'] = np.NAN

                # calculate features
                tmp_df = calc_fixation_features_tw(df_fixations_experiment, tw, t, r, p)
                all_exp_fixations_features = pd.concat([all_exp_fixations_features, tmp_df], axis=0)

                tmp_df = calc_pupil_features_tw(df_pupillometry_experiment, tw, t, r, p)
                all_exp_pupil_features = pd.concat([all_exp_pupil_features, tmp_df], axis=0)
            except FileNotFoundError:
                print("setting does not exist")

all_exp_fixations_features.to_csv(
    f"/Volumes/Data/chronopilot/Julia_study/features/all_exp_fixations_features_tw_{tw}.csv", index=False)
all_exp_pupil_features.to_csv(
    f"/Volumes/Data/chronopilot/Julia_study/features/all_exp_pupil_features_tw_{tw}.csv", index=False)
