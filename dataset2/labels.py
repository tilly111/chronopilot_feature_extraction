import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

input_file = os.path.join(const.BASE_DIR, "dataset2_RobotBehaviour", "Questionnaire_data1.xlsx")
output_dir = os.path.join(const.OUTPUT_DIR, "dataset2")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "labels.csv")

df = pd.read_excel(input_file)
df = df.rename(columns={"Index": "Participant"})

label_rows = []

valence_columns = [col for col in df.columns if str(col).count('.') == 2 and str(col).endswith(".3")]

for col in valence_columns:
    try:
        speed, robots, _ = col.split(".")
        speed = int(speed)
        robots = int(robots)
        task = (speed - 1) * 3 + robots

        for _, row in df.iterrows():
            valence = row[col]
            if pd.isna(valence):
                continue
            label = 1 if valence >= 6 else 0
            label_rows.append({
                "Participant": int(row["Participant"]),
                "Speed": speed,
                "Robots": robots,
                "Task": task,
                "Valence": valence,
                "Well-being": label
            })

    except Exception:
        continue

labels_df = pd.DataFrame(label_rows)
labels_df = labels_df.sort_values(by=["Participant", "Task"])
labels_df.to_csv(output_file, index=False)
