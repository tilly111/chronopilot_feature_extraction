tills_mbp = True

if tills_mbp:
    OUTPUT_DIR = "agg_data"  # Path to base folder for the output files
    BASE_DIR = "/Users/mariya_ty/Desktop/BA projekt"  # Path to the base folder for the input files
    FILTERED_DIR = "filtered_data"  # Path to the base folder for the filtered files
    # Window settings in samples (72 sec window, 20 sec step)
    INTERVAL = 72
    STEP = 20
else:
    OUTPUT_DIR = "agg_data"  # TODO check if this is correct
    BASE_DIR = "BA projekt"  # TODO check if this is correct
    FILTERED_DIR = "filtered_data"  # TODO check if this is correct
    INTERVAL = 72
    STEP = 20
    
