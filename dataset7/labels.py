import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

# Define file paths
input_file = os.path.join(const.BASE_DIR, "dataset7_Shui2021/Psychol_Rec/DRM.xlsx")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset7")
output_file = os.path.join(output_folder, "labels.csv")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the dataset
df = pd.read_excel(input_file)

# Define PANAS columns
positive_affect_cols = ['PANAS_3', 'PANAS_5', 'PANAS_7', 'PANAS_8', 'PANAS_10']
negative_affect_cols = ['PANAS_1', 'PANAS_2', 'PANAS_4', 'PANAS_6', 'PANAS_9']

# Calculate Positive and Negative Affect Scores
df['Positive_Affect'] = df[positive_affect_cols].sum(axis=1)
df['Negative_Affect'] = df[negative_affect_cols].sum(axis=1)

# Compute Composite Score
df['Composite_Score'] = df['Positive_Affect'] - df['Negative_Affect']

# Define the 'Classification' column based on Composite Score
df['Classification'] = df['Composite_Score'].apply(lambda x: 'Well-being' if x > 0 else 'Not Well-being')

# Define the Simplified Valence Classification
df['Simplified_Valence_Classification'] = df['Valence'].apply(
    lambda x: 'Well-being' if x > 3 else 'Not Well-being'
)

# Extract Participant and Task from "Event ID"
df['Participant'] = df['Event ID'].str.split('-').str[0]
df['Task'] = df['Event ID'].str.split('-').str[2]

# Extract Date from "Event ID"
df['Date'] = df['Event ID'].str.split('-').str[1]

# Map "well-being1" based on PANAS Classification (0 = Not Well-being, 1 = Well-being)
df['well-being1'] = df['Classification'].apply(lambda x: 1 if x == 'Well-being' else 0)

# Map "well-being2" based on Simplified Valence Classification (0 = Not Well-being, 1 = Well-being)
df['well-being2'] = df['Simplified_Valence_Classification'].apply(lambda x: 1 if x == 'Well-being' else 0)

# Retain only rows where 'well-being1' equals 'well-being2' and create a copy to avoid warnings
df_filtered = df[df['well-being1'] == df['well-being2']].copy()

# Create a single 'Well-Being' column
df_filtered.loc[:, 'Well-Being'] = df_filtered['well-being1']

# Retain only the required columns
label_data = df_filtered[['Participant', 'Task', 'Date', 'Well-Being', 'Valence', 'Arousal', 'Composite_Score']]

# Save the resulting dataframe to a CSV file
label_data.to_csv(output_file, index=False)

print(f"Filtered labels file with 'Well-Being' and 'Date' saved at: {output_file}")
