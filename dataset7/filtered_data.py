import pandas as pd
import os
from datetime import datetime

# Allgemeine Variablen und Pfade
base_folder = "dataset7_Shui2021"
output_folder_base = "filtered_data"
os.makedirs(output_folder_base, exist_ok=True)

# DRM-Datei laden
psychol_file_path = os.path.join(base_folder, "Psychol_Rec/DRM.xlsx")
drm_data = pd.read_excel(psychol_file_path)

# Funktion zur Verarbeitung eines Bereichs
def process_event_range(event_prefix, phys_folder, output_subfolder):
    event_rows = drm_data[drm_data['Event ID'].str.match(fr'^{event_prefix}\d{{3}}-')]
    output_folder = os.path.join(output_folder_base, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    if event_rows.empty:
        print(f"Keine passenden Event-IDs für Ordner '{phys_folder}' gefunden.")
        return

    print(f"Found {len(event_rows)} events starting with '{event_prefix}XXX-'.")

    for _, event_row in event_rows.iterrows():
        target_event_id = event_row['Event ID']
        interval = event_row['start | end time']

        # Start- und Endzeiten des Intervalls extrahieren und bereinigen
        start_time_str = interval.split('|')[0].strip()
        end_time_str = interval.split('|')[1].strip()

        try:
            event_date = f"2019-{target_event_id.split('-')[1][:2]}-{target_event_id.split('-')[1][2:]}"
            start_time = datetime.strptime(event_date + " " + start_time_str, "%Y-%m-%d %H:%M")
            end_time = datetime.strptime(event_date + " " + end_time_str, "%Y-%m-%d %H:%M")
        except ValueError as e:
            print(f"Fehler bei der Verarbeitung von Event ID {target_event_id}: {e}")
            continue

        print(f"Gefundenes Intervall für Event ID {target_event_id}: {start_time} - {end_time}")

        # Dateien im Ordner durchsuchen
        event_folder = os.path.join(base_folder, phys_folder, target_event_id.split('-')[0])
        if not os.path.exists(event_folder):
            print(f"Kein Ordner gefunden für Event ID {target_event_id}: {event_folder}")
            continue

        def get_matching_file(folder, target_start, target_end):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(".csv") and "_PPG" in file:
                        parts = file.replace(".csv", "").split("_")
                        if len(parts) < 2:
                            continue
                        try:
                            file_start = datetime.strptime(parts[0], "%Y%m%d%H%M%S")
                            file_end = datetime.strptime(parts[1], "%Y%m%d%H%M%S")
                        except ValueError as e:
                            print(f"Fehler beim Verarbeiten des Dateinamens {file}: {e}")
                            continue

                        if target_start >= file_start and target_end <= file_end:
                            return os.path.join(root, file)
            return None

        matching_file = get_matching_file(event_folder, start_time, end_time)

        if not matching_file:
            print(f"Keine passenden Dateien für Event ID {target_event_id} im Ordner '{event_folder}' gefunden.")
            continue

        print(f"Gefundene Datei für Event ID {target_event_id}: {matching_file}")

        try:
            data = pd.read_csv(matching_file)
            if data.empty:
                print(f"Die Datei {matching_file} ist leer und wird übersprungen.")
                continue
        except pd.errors.EmptyDataError:
            print(f"Die Datei {matching_file} ist leer oder ungültig und wird übersprungen.")
            continue
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {matching_file}: {e}")
            continue

        if 'csv_time_PPG' in data.columns:
            data['csv_time_PPG'] = pd.to_datetime(data['csv_time_PPG'], format="%d-%b-%Y %H:%M:%S", errors='coerce')

            filtered_data = data[
                (data['csv_time_PPG'] >= start_time) &
                (data['csv_time_PPG'] <= end_time)
            ]

            if filtered_data.empty:
                print(f"Keine Daten im gewünschten Intervall für Event ID {target_event_id}.")
                continue

            output_file_path = os.path.join(output_folder, f"filtered_{target_event_id}_{os.path.basename(matching_file)}")
            filtered_data.to_csv(output_file_path, index=False)
            print(f"Gefilterte Daten für Event ID {target_event_id} wurden gespeichert in: {output_file_path}")

# Bereiche verarbeiten
process_event_range("3", "Physiol_Rec 3", "Physiol_Rec_3")
process_event_range("1", "Physiol_Rec 1", "Physiol_Rec_1")
process_event_range("2", "Physiol_Rec 2", "Physiol_Rec_2")
