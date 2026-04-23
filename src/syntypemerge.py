import pandas as pd
import sys
import os
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import project-specific helpers, model wrappers, and configuration
from config     import *
from utils      import *

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

# Print a formatted header indicating the start of this processing stage
printHeader(f"Merging Generated Type Classifications")

# To track time.
start_time = time.time()






# ------------------------------------------------------------------------------
# Load Data.
# ------------------------------------------------------------------------------

# Merging Generated Classes.
if len(inputFileClassificationTypeMerged) == 0:
    log(f"No '{csvFileFormat}' files with prefix " + 
        f"'{outputFileNameClassificationTypeFormattedPrefix}' found in " + 
        f"'{outputFolderNameFormatted}'")
else:
    # Read and merge CSV files.
    dataframes = []
    log(f"Reading {len(inputFileClassificationTypeMerged)} files...")
    with newProgress() as progress:
        task = newTask(progress, len(inputFileClassificationTypeMerged), 
            "Processing Files")

        for file in sorted(inputFileClassificationTypeMerged):
            dataframes.append(pd.read_csv(file))
            progress.update(task, advance = 1)

        progress.refresh()

    # Read and merge CSV files.
    log("Files read.")

    log("Merging Data...")
    merged_df = pd.concat(dataframes, ignore_index = True)
    merged_df = merged_df.reset_index(drop = True)
    log("Data merged.")

    # --------------------------------------------------------------------------
    # Persist transformed data to disk.
    # --------------------------------------------------------------------------

    # Write merged CSV
    writeCSV(merged_df, outputFileClassificationTypeMerged)

    log(f"Merged {len(inputFileClassificationTypeMerged)} files into " + 
        f"'{os.path.basename(outputFileClassificationTypeMerged)}'")






# For time tracking.
minutes         = int((time.time() - start_time) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Data Merged [Minutes: {minutes}]")