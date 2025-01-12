import pandas as pd
import os

# Function to remove nested brackets
def remove_nested_brackets(df):
    """
    Removes nested brackets like [[value]] from a DataFrame and replaces them with simple values.
    """
    def unwrap(value):
        if isinstance(value, list) and len(value) > 0:
            return value[0] if isinstance(value[0], (float, int)) else value
        elif isinstance(value, str) and value.startswith('[[') and value.endswith(']]'):
            # Remove double brackets from strings like "[[123.45]]"
            try:
                return float(value.strip('[]'))
            except ValueError:
                return value
        return value  # Return the original value if it's not nested
    
    return df.applymap(unwrap)

# Function to clean all matching files in the folder and subfolders
def clean_ecg_features_in_all_subfolders(base_folder):
    """
    Searches all subfolders of 'base_folder' for files starting with 'ecg_features',
    cleans them, and saves them back to the original files.
    """
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.startswith("ecg_features") and file.endswith(".csv"):  # Match files starting with 'ecg_features'
                file_path = os.path.join(root, file)
                try:
                    print(f"Cleaning file: {file_path}")
                    
                    # Load the file
                    df = pd.read_csv(file_path)
                    
                    # Remove nested brackets
                    clean_df = remove_nested_brackets(df)
                    
                    # Overwrite the file
                    clean_df.to_csv(file_path, index=False)
                    print(f"File successfully cleaned and saved: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

# Main execution
if __name__ == "__main__":
    base_folder = "agg_data"  # Path to the main folder
    clean_ecg_features_in_all_subfolders(base_folder)
