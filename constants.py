SCREAM_DATA_PATH = "/Volumes/Data/chronopilot/scream_experiment/"
SUBJECTS_STUDY_1 = [2]  # TODO incomplete
SUBJECTS_STUDY_2 = [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 16, 18, 23, 25, 26, 27, 29, 31, 32, 33, 41, 42, 45, 46, 47]

ALL_PPG_FEATURES_HEARTPY = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2', 's',
                            'sd1/sd2', 'breathingrate']
ALL_PPG_FEATURES_NEUROKIT = ['PPG_Rate_Mean', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1',
                             'HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD',
                             'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN',
                             'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN',
                             'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI',
                             'HRV_TINN', 'HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF',
                             'HRV_TP', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1',
                             'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 'HRV_CVI',
                             'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_PAS',
                             'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
                             'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
                             'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
                             'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                             'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                             'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
                             'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment',
                             'HRV_DFA_alpha2', 'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak',
                             'HRV_MFDFA_alpha2_Mean', 'HRV_MFDFA_alpha2_Max',
                             'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
                             'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment',
                             'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn',
                             'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
ALL_EDA_FEATURES = ["SCR_Peaks_N",
                    "SCR_Peaks_Amplitude_Mean",
                    "EDA_Tonic_SD",
                    "EDA_Sympathetic",
                    "EDA_SympatheticN",
                    "EDA_Autocorrelation"]
ALL_TMP_FEATURES = ["mean_tmp_dif", "mean_th", "mean_t1", "gradient", "psd_power"]
