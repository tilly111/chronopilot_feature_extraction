import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

# Load the NASA TLX data file
file_path = os.path.join(const.BASE_DIR, 'dataset1_SenseCobot/Additional_Information/NASA_TLX.csv')
nasa_tlx_data = pd.read_csv(file_path)


# Remove extra spaces from column names
nasa_tlx_data.columns = nasa_tlx_data.columns.str.strip()

# Drop empty rows
nasa_tlx_data = nasa_tlx_data.dropna(how='all')

# Function to calculate the stress label for a task
def compute_stress_label(row, task_num):
    total_score = row[f'A_TASK {task_num}'] + row[f'B_TASK {task_num}']  # Sum A and B scores
    return 1 if total_score >= 7 else 0  # Label as stress if the total is 7 or more

# Prepare a new structured dataset
data = []

tasks = range(1, 6)  # Tasks 1 to 5

for participant_id, row in enumerate(nasa_tlx_data.iterrows(), start=1):
    row = row[1]  # Extract the row (second value in the tuple from iterrows)
    for task in tasks:
        a_score = row[f'A_TASK {task}']  # Score for A_TASK
        b_score = row[f'B_TASK {task}']  # Score for B_TASK
        stress_label = compute_stress_label(row, task)  # Calculate stress label
        well_being_label = 0 if stress_label == 1 else 1  # Calculate well-being label

        data.append({
            'Participant': participant_id,  # Participant IDs start at 1
            'Task': task,
            'Stress': stress_label,  # Stress label (binary)
            'Well-Being': well_being_label,  # Well-being label (binary)
            'A_Score': a_score,  # Score from A_TASK
            'B_Score': b_score   # Score from B_TASK
        })

# Create a new DataFrame from the processed data
processed_data = pd.DataFrame(data)

# Save the processed data to the desired folder
output_folder = 'agg_data/dataset1'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
output_file = os.path.join(output_folder, 'labels.csv')

os.makedirs(output_folder, exist_ok=True)

print(f'The processed data has been saved to: {output_file}')
