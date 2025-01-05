import os
import pandas as pd

# Define file paths
database_file = "dataset5_Markova2021/CLAS/Documentation/DataBaseDescription.xlsx"
setup1_file = "dataset5_Markova2021/CLAS/Documentation/Setup1.xlsx"
setup2_file = "dataset5_Markova2021/CLAS/Documentation/Setup2.xlsx"

# Define output directory and file paths
output_dir = "agg_data/dataset5"
os.makedirs(output_dir, exist_ok=True)  

output_labels_file = os.path.join(output_dir, "labels.csv")
output_setup1_file = os.path.join(output_dir, "setup1_details.csv")
output_setup2_file = os.path.join(output_dir, "setup2_details.csv")

# Read input data
stset1_data = pd.read_excel(database_file, sheet_name="Stset1")  
stimtag_data = pd.read_excel(database_file, sheet_name="StimTag")  
stset2_data = pd.read_excel(database_file, sheet_name="Stset2")  

# Process Setup1
merged_data1 = pd.merge(setup1_data, stset1_data, left_on="block", right_on="BlockID", how="inner")  
filtered_data1 = merged_data1[merged_data1["StimuliTag"].str.lower() == "yes"]
merged_with_stimtag1 = pd.merge(filtered_data1, stimtag_data[["StName", "Valence"]], on="StName", how="inner") 
merged_with_stimtag1["Well-Being"] = merged_with_stimtag1["Valence"].apply(lambda x: 0 if x < 5 else 1) 
output_grouped1 = merged_with_stimtag1[["block", "Well-Being"]].drop_duplicates() 
participants1 = list(range(1, 12))  # Participants 1 to 11
output_with_participants1 = output_grouped1.assign(key=1).merge(
    pd.DataFrame({"Participant": participants1, "key": 1}), on="key"
).drop("key", axis=1)  

# Process Setup2
merged_data2 = pd.merge(setup2_data, stset2_data, left_on="block", right_on="BlockID", how="inner")  
filtered_data2 = merged_data2[merged_data2["StimuliTag"].str.lower() == "yes"]  
merged_with_stimtag2 = pd.merge(filtered_data2, stimtag_data[["StName", "Valence"]], on="StName", how="inner")  
merged_with_stimtag2["Well-Being"] = merged_with_stimtag2["Valence"].apply(lambda x: 0 if x < 5 else 1)  
output_grouped2 = merged_with_stimtag2[["block", "Well-Being"]].drop_duplicates()  
participants2 = list(range(12, 61))  # Participants 12 to 60
output_with_participants2 = output_grouped2.assign(key=1).merge(
    pd.DataFrame({"Participant": participants2, "key": 1}), on="key"
).drop("key", axis=1)  # Add Participant column

# Combine results for labels.csv
final_output = pd.concat([output_with_participants1, output_with_participants2]).rename(
    columns={"block": "Task"}  # Rename "block" to "Task"
).sort_values(by=["Participant", "Task"]).reset_index(drop=True)  # Combine and sort by Participant and Task

final_output = final_output[["Participant", "Task", "Well-Being"]]

final_output.to_csv(output_labels_file, index=False)

merged_with_stimtag1[["block", "stimulus", "StName", "StimuliTag", "Valence", "Well-Being"]].to_csv(output_setup1_file, index=False)
merged_with_stimtag2[["block", "stimulus", "StName", "StimuliTag", "Valence", "Well-Being"]].to_csv(output_setup2_file, index=False)
