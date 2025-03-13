tills_mbp = True

if tills_mbp:
    OUTPUT_DIR = "/Volumes/Data/chronopilot/external_datasets/agg_data"  # Path to base folder for the output files
    BASE_DIR = "/Volumes/Data/chronopilot/external_datasets"  # Path to the base folder for the input files
    FILTERED_DIR = "/Volumes/Data/chronopilot/external_datasets/filtered_data"  # Path to the base folder for the filtered files
else:
    OUTPUT_DIR = "agg_data"  # TODO check if this is correct
    BASE_DIR = ""  # TODO check if this is correct
    FILTERED_DIR = "filtered_data"  # TODO check if this is correct