import warnings
import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_pupillometry_features(bsl_data, exp_data, slicing) -> tuple[list, list, list]:
    '''
    features: norm_pos_x, norm_pos_y, diameter0_2d, diameter1_2d, diameter0_3d, diameter1_3d
    :param bsl_data: Baseline dataframe with columns: norm_pos_x, norm_pos_y, diameter0_2d, diameter1_2d, diameter0_3d, diameter1_3d
    :param exp_data: Experiment dataframe with columns: norm_pos_x, norm_pos_y, diameter0_2d, diameter1_2d, diameter0_3d, diameter1_3d
    :param slicing: True if the data are sliced to 1 min pieces
    :return: bsl_features, exp_features, sub_features
    '''
    # bsl
    bsl_df = pd.DataFrame()
    bsl_df['distance'] = distance(bsl_data["norm_pos_x"], bsl_data["norm_pos_y"], bsl_data["norm_pos_x"].shift(-1), bsl_data["norm_pos_y"].shift(-1))

    bsl_min_speed = bsl_df['distance'].min()
    bsl_max_speed = bsl_df['distance'].max()
    bsl_mean_speed = bsl_df['distance'].mean()
    bsl_std_speed = bsl_df['distance'].std()

    # Create feature matrix
    X = bsl_data[['norm_pos_x', 'norm_pos_y']].values

    # Silhouette Method -> starts at 2
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Method')
    # plt.show()

    bsl_number_clusters = np.max(silhouette_scores) / 120  # max silhouette score is the best fitting of clusters
    # bsl_number_clusters = np.nan

    # TODO we need to interpolate the diameter values to do not have NaNs in the data
    for col in ['diameter0_2d', 'diameter1_2d', 'diameter0_3d', 'diameter1_3d']:
        bsl_data[col] = bsl_data[col].interpolate()
        if np.isnan(bsl_data[col].iloc[0]):
            # bsl_data[col].iloc[0] = bsl_data[col].iloc[1]
            bsl_data.loc[0, col] = float(bsl_data[col].iloc[1])  # TODO does this throw an error

    # diameter change
    bsl_df['diameter2d_change'] = distance(bsl_data["diameter0_2d"], bsl_data["diameter1_2d"], bsl_data["diameter0_2d"].shift(-1), bsl_data["diameter1_2d"].shift(-1))
    bsl_df['diameter3d_change'] = distance(bsl_data["diameter0_3d"], bsl_data["diameter1_3d"], bsl_data["diameter0_3d"].shift(-1), bsl_data["diameter1_3d"].shift(-1))

    bsl_min_diameter2d = bsl_df['diameter2d_change'].min()
    bsl_max_diameter2d = bsl_df['diameter2d_change'].max()
    bsl_mean_diameter2d = bsl_df['diameter2d_change'].mean()

    bsl_min_diameter3d = bsl_df['diameter3d_change'].min()
    bsl_max_diameter3d = bsl_df['diameter3d_change'].max()
    bsl_mean_diameter3d = bsl_df['diameter3d_change'].mean()

    bsl_features = [bsl_min_speed, bsl_max_speed, bsl_mean_speed, bsl_std_speed, bsl_number_clusters, bsl_min_diameter2d, bsl_max_diameter2d, bsl_mean_diameter2d, bsl_min_diameter3d, bsl_max_diameter3d, bsl_mean_diameter3d]

    # exp
    exp_df = pd.DataFrame()
    exp_df['distance'] = distance(exp_data["norm_pos_x"], exp_data["norm_pos_y"], exp_data["norm_pos_x"].shift(-1),
                                  exp_data["norm_pos_y"].shift(-1))

    exp_min_speed = exp_df['distance'].min()
    exp_max_speed = exp_df['distance'].max()
    exp_mean_speed = exp_df['distance'].mean()

    # Create feature matrix
    X = exp_data[['norm_pos_x', 'norm_pos_y']].values

    # Silhouette Method -> starts at 2
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Method')
    # plt.show()

    if slicing:
        exp_number_clusters = np.max(silhouette_scores) / int(60)  # max silhouette score is the best fitting of clusters
    else:
        exp_number_clusters = np.max(silhouette_scores) / int(
            exp_data["timestamp"].iloc[-1])  # max silhouette score is the best fitting of clusters
    # exp_number_clusters = np.nan

    # TODO we need to interpolate the diameter values to do not have NaNs in the data
    for col in ['diameter0_2d', 'diameter1_2d', 'diameter0_3d', 'diameter1_3d']:
        exp_data[col] = exp_data[col].interpolate()
        if np.isnan(exp_data[col].iloc[0]):
            # bsl_data[col].iloc[0] = bsl_data[col].iloc[1]
            exp_data.loc[0, col] = float(exp_data[col].iloc[1])  # TODO does this throw an error
        # print(f"bsl: {exp_data[col].isnull().sum()}")

    # diameter change
    exp_df['diameter2d_change'] = distance(exp_data["diameter0_2d"], exp_data["diameter1_2d"], exp_data["diameter0_2d"].shift(-1), exp_data["diameter1_2d"].shift(-1))
    exp_df['diameter3d_change'] = distance(exp_data["diameter0_3d"], exp_data["diameter1_3d"], exp_data["diameter0_3d"].shift(-1), exp_data["diameter1_3d"].shift(-1))

    exp_min_diameter2d = exp_df['diameter2d_change'].min()
    exp_max_diameter2d = exp_df['diameter2d_change'].max()
    exp_mean_diameter2d = exp_df['diameter2d_change'].mean()

    exp_min_diameter3d = exp_df['diameter3d_change'].min()
    exp_max_diameter3d = exp_df['diameter3d_change'].max()
    exp_mean_diameter3d = exp_df['diameter3d_change'].mean()

    exp_features = [exp_min_speed, exp_max_speed, exp_mean_speed, exp_number_clusters, exp_min_diameter2d, exp_max_diameter2d, exp_mean_diameter2d, exp_min_diameter3d, exp_max_diameter3d, exp_mean_diameter3d]

    return bsl_features, exp_features, [x - y for x, y in zip(exp_features, bsl_features)]


def calc_fixations_features(bsl_data, exp_data, slicing) -> tuple[list, list, list]:
    '''
    features: norm_pos_x, norm_pos_y, dispersion, duration
    :param bsl_data: Baseline dataframe with columns: norm_pos_x, norm_pos_y, dispersion, duration
    :param exp_data: Experiment dataframe with columns: norm_pos_x, norm_pos_y, dispersion, duration
    :param slicing: True if the data are sliced to 1 min pieces
    :return: bsl_features, exp_features, sub_features
    '''
    # bsl
    bsl_df = pd.DataFrame()
    bsl_df['distance'] = distance(bsl_data["norm_pos_x"], bsl_data["norm_pos_y"], bsl_data["norm_pos_x"].shift(-1),
                                  bsl_data["norm_pos_y"].shift(-1))

    bsl_min_speed = bsl_df['distance'].min()
    bsl_max_speed = bsl_df['distance'].max()
    bsl_mean_speed = bsl_df['distance'].mean()

    bsl_min_duration = bsl_data['duration'].min()
    bsl_max_duration = bsl_data['duration'].max()
    bsl_mean_duration = bsl_data['duration'].mean()

    # Create feature matrix
    X = bsl_data[['norm_pos_x', 'norm_pos_y']].values

    # Silhouette Method -> starts at 2
    try:
        silhouette_scores = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        bsl_number_clusters = np.max(silhouette_scores) / 120  # max silhouette score is the best fitting of clusters
    except ValueError:
        print("ValueError in cluster calculatation")
        bsl_number_clusters = np.nan

    bsl_min_dispersion = bsl_data["dispersion"].min()
    bsl_max_dispersion = bsl_data["dispersion"].max()
    bsl_mean_dispersion = bsl_data["dispersion"].mean()

    # exp
    exp_df = pd.DataFrame()
    exp_df['distance'] = distance(exp_data["norm_pos_x"], exp_data["norm_pos_y"], exp_data["norm_pos_x"].shift(-1),
                                  exp_data["norm_pos_y"].shift(-1))

    exp_min_speed = exp_df['distance'].min()
    exp_max_speed = exp_df['distance'].max()
    exp_mean_speed = exp_df['distance'].mean()

    # Create feature matrix
    X = exp_data[['norm_pos_x', 'norm_pos_y']].values

    # Silhouette Method -> starts at 2
    try:
        silhouette_scores = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        if slicing:
            exp_number_clusters = np.max(silhouette_scores) / int(60)
        else:
            exp_number_clusters = np.max(silhouette_scores) / int(
                exp_data["timestamp"].iloc[-1])  # max silhouette score is the best fitting of clusters
    except ValueError:
        print("ValueError in cluster calculatation")
        exp_number_clusters = np.nan

    exp_min_dispersion = exp_data["dispersion"].min()
    exp_max_dispersion = exp_data["dispersion"].max()
    exp_mean_dispersion = exp_data["dispersion"].mean()

    exp_min_duration = exp_data['duration'].min()
    exp_max_duration = exp_data['duration'].max()
    exp_mean_duration = exp_data['duration'].mean()

    bsl_features = [bsl_min_speed, bsl_max_speed, bsl_mean_speed, bsl_number_clusters, bsl_min_dispersion, bsl_max_dispersion, bsl_mean_dispersion, bsl_min_duration, bsl_max_duration, bsl_mean_duration]
    exp_features = [exp_min_speed, exp_max_speed, exp_mean_speed, exp_number_clusters, exp_min_dispersion, exp_max_dispersion, exp_mean_dispersion, exp_min_duration, exp_max_duration, exp_mean_duration]

    return bsl_features, exp_features, [x - y for x, y in zip(exp_features, bsl_features)]


# TODO make Pupil data -> pupil diameter positively correlated with task difficulty
def calc_pupil_features_tw(exp_data, time_window, time, robot, participant) -> pd.DataFrame:
    # data frame
    feature_df = pd.DataFrame(columns=["time", "robot", "participant", "slice"] + ["pupil_diameter0_2d_mean",
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

    # time window should be in seconds
    print(f"shape: {exp_data.shape[0]}")
    # if data are empty
    if exp_data.shape[0] == 0:
        feature_df.loc[0] = [time, robot, participant, 0] + [np.nan] * 10
        return
    n_time_window = int(np.round(exp_data["timestamp"].iloc[-1] / time_window))
    print(f"{n_time_window} = {exp_data['timestamp'].iloc[-1]} / {(time_window)}")
    for i in range(n_time_window):
        slice = i * time_window
        # bsl_tw = bsl_data.loc[(df_pupillometry_baseline["timestamp"] >= slice) & (df_pupillometry_baseline["timestamp"] < slice + time_window)]
        exp_tw = exp_data.loc[
            (exp_data["timestamp"] >= slice) & (exp_data["timestamp"] < slice + time_window)]
        print(
            f"Time slice: {slice}, {slice + time_window}; {exp_tw.shape[0]} values within the confidence threshold")
        if exp_tw.shape[0] < 10:
            print(f"haaa {exp_tw.shape[0]}")
            print("no values within the confidence threshold")
            continue

        # IPA (Index of pupillary activity): IPA0, mean pupil diameter, pupil0 deviation, max pupil0 diameter,
        #   IPA1, mean pupil1 diameter, pupil1 deviation, max pupil diameter1 diameter
        #   0 = left, 1 = right -> bei uns probably anders rum?! aber ist das wichtig?
        dia_0_2d_max = exp_tw["diameter0_2d"].dropna().max()
        dia_1_2d_max = exp_tw["diameter1_2d"].dropna().max()
        dia_0_2d_mean = exp_tw["diameter0_2d"].dropna().mean()
        dia_1_2d_mean = exp_tw["diameter1_2d"].dropna().mean()
        dia_0_2d_dev = exp_tw["diameter0_2d"].dropna().std()
        dia_1_2d_dev = exp_tw["diameter1_2d"].dropna().std()

        dia_0_3d_max = exp_tw["diameter0_3d"].dropna().max()
        dia_1_3d_max = exp_tw["diameter1_3d"].dropna().max()
        dia_0_3d_mean = exp_tw["diameter0_3d"].dropna().mean()
        dia_1_3d_mean = exp_tw["diameter1_3d"].dropna().mean()
        dia_0_3d_dev = exp_tw["diameter0_3d"].dropna().std()
        dia_1_3d_dev = exp_tw["diameter1_3d"].dropna().std()

        def ipa(d):
            '''
            Taken from "The Index of Pupillary Activity" by A. T. Duchowski et al. (2018)
            :param d: pupil diameter signal
            :return:
            '''

            def modmax(d):
                # compute signal
                m = [0.0] * len(d)
                for i in range(len(d)):
                    m[i] = np.fabs(d[i])
                # if value is larger than both neighbours, and strictly
                # larger than either, then it is a local maximum
                t = [0.0] * len(d)
                for i in range(len(d)):
                    ll = m[i - 1] if i >= 1 else m[i]
                    oo = m[i]
                    rr = m[i + 1] if i < len(d) - 2 else m[i]
                    if (ll <= oo and oo >= rr) and (ll < oo or oo > rr):
                        # compute magnitude
                        t[i] = np.sqrt(d[i] ** 2)
                    else:
                        t[i] = 0.0
                return t

            # obtain 2-level DWT of pupil diameter signal d
            try:
                (cA2, cD2, cD1) = pywt.wavedec(d[:, 1], 'sym16', 'per', level=2)
            except ValueError:
                return np.nan
            # get signal duration (in seconds)
            tt = float(d[-1, 0] - d[0, 0])
            if tt == 0:
                return np.nan
            # normalize by 1 / 2j, j = 2
            cA2[:] = [x / np.sqrt(4.0) for x in cA2]
            cD1[:] = [x / np.sqrt(2.0) for x in cD1]
            cD2[:] = [x / np.sqrt(4.0) for x in cD2]

            # detect modulus maxima , see listing 2
            cD2m = modmax(cD2)

            lambda_univ = np.std(cD2m) * np.sqrt(2.0 * np.log2(len(cD2m)))
            cD2t = pywt.threshold(cD2m, lambda_univ, mode="hard")
            # compute IPA
            ctr = 0
            for ii in range(len(cD2t)):
                if np.fabs(cD2t[ii]) > 0:
                    ctr += 1
                IPA = float(ctr) / tt
            return IPA

        dia_0_2d_ipa = ipa(exp_tw[["timestamp", "diameter0_2d"]].dropna().to_numpy())
        dia_1_2d_ipa = ipa(exp_tw[["timestamp", "diameter1_2d"]].dropna().to_numpy())
        dia_0_3d_ipa = ipa(exp_tw[["timestamp", "diameter0_3d"]].dropna().to_numpy())
        dia_1_3d_ipa = ipa(exp_tw[["timestamp", "diameter1_3d"]].dropna().to_numpy())

        feature_df.loc[i] = [time, robot, participant, i] + [dia_0_2d_mean, dia_0_2d_max, dia_0_2d_dev, dia_0_2d_ipa,
                                                             dia_1_2d_mean, dia_1_2d_max, dia_1_2d_dev, dia_1_2d_ipa,
                                                             dia_0_3d_mean, dia_0_3d_max, dia_0_3d_dev, dia_0_3d_ipa,
                                                             dia_1_3d_mean, dia_1_3d_max, dia_1_3d_dev, dia_1_3d_ipa]

    return feature_df


def calc_fixation_features_tw(exp_data, time_window, time, robot, participant) -> pd.DataFrame:
    '''

    :param exp_data:
    :param time_window:
    :param time: time should be in seconds!
    :param robot:
    :param participant:
    :return:
    '''
    # data frame
    feature_df = pd.DataFrame(
        columns=["time", "robot", "participant", "slice"] + ["fixation_frequency", "fixation_duration_mean",
                                                             "fixation_duration_max", "fixation_dispersion_mean",
                                                             "fixation_dispersion_max", "saccade_frequency",
                                                             "saccade_durations_mean", "saccade_durations_max",
                                                             "saccade_speed_mean", "saccade_speed_max"])
    # if data are empty
    if exp_data.shape[0] == 0:
        feature_df.loc[0] = [time, robot, participant, 0] + [np.nan] * 10
        return
    n_time_window = int(np.round(exp_data["timestamp"].iloc[-1] / time_window))
    print(f"{n_time_window} = {exp_data['timestamp'].iloc[-1]} / {(time_window)}")
    for i in range(n_time_window):
        slice = i * time_window
        # bsl_tw = bsl_data.loc[(df_pupillometry_baseline["timestamp"] >= slice) & (df_pupillometry_baseline["timestamp"] < slice + time_window)]
        exp_tw = exp_data.loc[(exp_data["timestamp"] >= slice) & (exp_data["timestamp"] < slice + time_window)]
        print(f"Time slice: {slice}, {slice + time_window}; {exp_tw.shape[0]} values within the confidence threshold")
        if exp_tw.shape[0] < 10:
            print(f"haaa {exp_tw.shape[0]}")
            print("no values within the confidence threshold")
            continue
        # calculate eye-movement features
        # fixation frequency = total number of eye fixations on a target stimuli; i.e., fixation ids?
        fixation_frequency = exp_tw["fixation id"].nunique() / time_window
        # mean fixation duration
        fixation_duration_mean = exp_tw["duration"].mean()
        # max fixation duration
        fixation_duration_max = exp_tw["duration"].max()

        # calculate dispersion
        fixation_dispersion_mean = np.nanmean(exp_tw["dispersion"])
        fixation_dispersion_max = np.nanmax(exp_tw["dispersion"])

        # mean saccade frequency
        from scipy.signal import savgol_filter
        x_norm = savgol_filter(exp_tw["norm_pos_x"], 5, 3)  # TODO find hyperparameters
        y_norm = savgol_filter(exp_tw["norm_pos_y"], 5, 3)

        velocity_x = np.diff(x_norm) / np.diff(exp_tw["timestamp"])
        velocity_y = np.diff(y_norm) / np.diff(exp_tw["timestamp"])
        velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        # saccade threshold; TODO what is a approriate threshold?
        saccade_threshold = 0.3
        saccades = velocity > saccade_threshold
        saccade_count = np.sum(saccades)

        print(saccade_count)

        saccade_frequency = saccade_count / time_window

        # Identify start and end of saccades
        saccade_start_indices = np.where((velocity[:-1] < saccade_threshold) & (velocity[1:] >= saccade_threshold))[
                                    0] + 1
        saccade_end_indices = np.where((velocity[:-1] >= saccade_threshold) & (velocity[1:] < saccade_threshold))[0] + 1

        # Calculate saccade durations
        saccade_durations = []
        for start, end in zip(saccade_start_indices, saccade_end_indices):
            duration = exp_tw["timestamp"].iloc[end] - exp_tw["timestamp"].iloc[start]
            saccade_durations.append(duration)

        if len(saccade_durations) == 0:
            print("no saccades found")
            saccade_durations_mean = np.nan
            saccade_durations_max = np.nan
        else:
            saccade_durations_mean = np.nanmean(saccade_durations)
            saccade_durations_max = np.nanmax(saccade_durations)

        if saccade_count < 1:
            saccade_speed_mean = np.nan
            saccade_speed_max = np.nan
        else:
            saccade_speed_mean = np.nanmean(velocity[saccades])
            saccade_speed_max = np.nanmax(velocity[saccades])

        feature_df.loc[i] = [time, robot, participant, i] + [fixation_frequency, fixation_duration_mean,
                                                             fixation_duration_max, fixation_dispersion_mean,
                                                             fixation_dispersion_max, saccade_frequency,
                                                             saccade_durations_mean, saccade_durations_max,
                                                             saccade_speed_mean, saccade_speed_max]

    return feature_df


# todo blink data?
