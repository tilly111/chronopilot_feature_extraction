# import csv
import pandas as pd
import numpy as np
# from IPython.display import display


class Data_Loader:

    def __init__(self, data_path, verbose=True):
        self.verbose = verbose

        self.data_path = data_path


    # load the corrected trajectory
    def load_trajectory(self, person, speed=1, number_robots=1, upper_bound=179.0):
        data_path_x = "data/Experiment1(robot behaviour)/Measurements_fixed/p_" + str(person) + "/" + str(
            speed) + "_" + str(number_robots) + "/X-CoordinatesWithoutMistakes.csv"
        data_path_y = "data/Experiment1(robot behaviour)/Measurements_fixed/p_" + str(person) + "/" + str(
            speed) + "_" + str(number_robots) + "/Y-CoordinatesWithoutMistakes.csv"

        df_x = pd.read_csv(data_path_x, sep=';', header=0, decimal=',')
        df_y = pd.read_csv(data_path_y, sep=';', header=0, decimal=',')

        # cut to slice_slow specified length i.e. 180 sec
        df_x = df_x.loc[df_x["TimeStamp"] <= upper_bound]
        df_y = df_y.loc[df_y["TimeStamp"] <= upper_bound]

        return df_x, df_y

    ### returns data frame
    # @speed can either be 1 (slow) or 2 (fast)
    # @number_robots can be 1 (1), 2(5), or 3(15)
    def load_data(self, person, speed=1, number_robots=1, data_type="X-Coordinates", labels=None, upper_bound=179.0, norm_method=None):
        # load correct data frame - ts
        data_path = "data/Experiment1(robot behaviour)/Measurements_fixed/p_" + str(person) + "/" + str(speed) + "_" + str(
            number_robots) + "/" + str(data_type) + ".csv"
        df_x = pd.read_csv(data_path, sep=';', header=0, decimal=',')

        # cut to slice_slow specified length i.e. 180 sec
        df_x["TimeStamp"] = df_x["TimeStamp"].clip(upper=upper_bound)

        # normalize the data using some normalization method
        if norm_method == "zscore":
            m = np.mean(df_x[data_type].to_numpy())
            std = np.std(df_x[data_type].to_numpy())
            df_x[data_type] = (df_x[data_type] - m)/std

        # hack for the button data which should be 0 or one?
        if data_type == "state":

            df_x.loc[df_x['state'] > 2, 'state'] = 1
            df_x.loc[df_x['state'] == 1, 'state'] = 0
            df_x.loc[df_x['state'] == 2, 'state'] = 1

        # load fitting label - perceived passage of time
        id = str(speed) + '.' + str(number_robots) + '.2'
        time_pass_label = labels.at[person - 1, id]
        # load arousal
        id = str(speed) + '.' + str(number_robots) + '.4'
        arousal_label = labels.at[person - 1, id]

        df_y = pd.DataFrame(data={'label_pass_time': [time_pass_label], 'label_arousal': [arousal_label]})

        return df_x, df_y

    # returns a nested list in the following entries
    # data[p][s][r][xx]; p = person-1, s = speed -1, r = number robots,
    # def load_all_RR(self):
    #     data = [[[[None for _ in range(2)] for _ in range(4)] for _ in range(3)] for _ in range(25)]
    #     for p in range(1, 26):
    #         for s in range(1, 3):
    #             for r in range(1, 4):
    #                 x, y = self.load_data(p, s, r, "RR")
    #                 data[p - 1][s - 1][r - 1][0] = x
    #                 data[p - 1][s - 1][r - 1][1] = y
    #
    #     return data
    #
    # def load_all_EDA(self):
    #     data = [[[[None for _ in range(2)] for _ in range(4)] for _ in range(3)] for _ in range(25)]
    #     for p in range(1, 26):
    #         for s in range(1, 3):
    #             for r in range(1, 4):
    #                 x, y = self.load_data(p, s, r, "EDA", 179.0)
    #                 data[p - 1][s - 1][r - 1][0] = x
    #                 data[p - 1][s - 1][r - 1][1] = y
    #
    #     return data

    def load_experiment1(self, kind="RR", norm_method=None):
        labels = pd.read_csv("data/Experiment1(robot behaviour)/Questionnaire_data.csv", sep=';', header=0, decimal=',')
        data = [[[[None for _ in range(2)] for _ in range(4)] for _ in range(3)] for _ in range(25)]
        for p in range(1, 26):
            for s in range(1, 3):
                for r in range(1, 4):
                    x, y = self.load_data(p, s, r, kind, labels, norm_method=norm_method)
                    data[p - 1][s - 1][r - 1][0] = x
                    data[p - 1][s - 1][r - 1][1] = y

        return data


    def load_cue(self, kind="h10_ecg", norm_method=None):
        labels = pd.read_csv("data/CUE_Experiment/Questionnaire_data.csv", header=0, sep=';')
        persons = ["FS_22", "JS_22", "KP_37", "NC_25", "NW_24", "RA_25", "SZ_30"]  # "AF_27", "MA_24", "AK_26"
        cues = ["cue_0", "cue_1", "cue_2", "cue_3"]
        data = [[[None for _ in range(2)] for _ in range(len(cues))] for _ in range(len(persons))]
        # for all subjects
        for p_i, p in enumerate(persons):
            # for all settings
            for c_i, c in enumerate(cues):
                data_path = "data/CUE_Experiment/" + p + "/" + c + "/" + kind + ".csv"
                x = pd.read_csv(data_path, header=None)
                x = x.rename(columns={0: 'TimeStamp', 1: kind})
                x["TimeStamp"] = x["TimeStamp"] - x["TimeStamp"][0]
                y = labels.loc[(labels['Initials'] == p) & (labels['Cue Type'] == c_i)]
                data[p_i-1][c_i-1][0] = x
                data[p_i-1][c_i-1][1] = y
        return data



    # Format: person x setting x 0 = data, 1 = label
    def load_helicopter(self, kind="RR", norm_method=None):
        labels = pd.read_csv("/Volumes/Data/chronopilot/helicopter/timings.csv", header=0)
        data = [[[None for _ in range(2)] for _ in range(4)] for _ in range(12)]
        # for all subjects
        for p in range(1, 13):
            # for all settings
            for s in range(1, 5):
                data_path = "/Volumes/Data/chronopilot/helicopter/Physiological/" + kind + "/subject" + str(p) + "-" + str(s) + "_" + kind + ".csv"
                x = pd.read_csv(data_path, header=0)
                y = labels.loc[(labels['ParticipantID'] == p) & (labels['Session'] == s)]
                data[p-1][s-1][0] = x
                data[p-1][s-1][1] = y
        return data
