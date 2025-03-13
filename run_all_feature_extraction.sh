# NOTE make sure you source correctly
source ~/PycharmProjects/ghent-colab/venv/bin/activate

echo $PWD

# dataset1
python chronopilot_feature_extraction/dataset1/ecg_features.py &
python chronopilot_feature_extraction/dataset1/eda_features.py &
python chronopilot_feature_extraction/dataset1/eeg_features.py &
python chronopilot_feature_extraction/dataset1/gsr_features.py &
python chronopilot_feature_extraction/dataset1/labels.py &
python chronopilot_feature_extraction/dataset1/tmp_features.py &

# dataset3
python chronopilot_feature_extraction/dataset3/ecg_features.py &
python chronopilot_feature_extraction/dataset3/gsr_features.py &
python chronopilot_feature_extraction/dataset3/labels.py &
python chronopilot_feature_extraction/dataset3/ppg_features.py &

# dataset4
python chronopilot_feature_extraction/dataset4/ecg_features.py &

# dataset5
python chronopilot_feature_extraction/dataset5/ecg_features.py &
python chronopilot_feature_extraction/dataset5/eda_features.py &
python chronopilot_feature_extraction/dataset5/labels.py &
python chronopilot_feature_extraction/dataset5/ppg_features.py &

# dataset7
python chronopilot_feature_extraction/dataset7/filtered_data.py &
python chronopilot_feature_extraction/dataset7/gsr_features.py &
python chronopilot_feature_extraction/dataset7/labels.py &
python chronopilot_feature_extraction/dataset7/ppg_features.py &

# dataset8
python chronopilot_feature_extraction/dataset8/ecg_features.py &
python chronopilot_feature_extraction/dataset8/eda_features.py &
python chronopilot_feature_extraction/dataset8/labels.py &

# dataset9
python chronopilot_feature_extraction/dataset9/dataset9.py

wait

deactivate