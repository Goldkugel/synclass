import sys
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config     import *
from utils      import *

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------


printHeader(f"Fomratting Classification of Synonyms")

# To track time.
start_time = time.time()

# ------------------------------------------------------------------------------
# Load Human Phenotype Ontology (HPO) data.
# ------------------------------------------------------------------------------

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassificationTypeFormatted)

# Load the dataset from a pickle file
classified    = readCSV(inputFileClassificationTypeFormatted)






classified[confidenceColumn] = [-1] * len(classified.index)

with newProgress() as progress:

    task = newTask(progress, len(classified.index), "Formatting Answers")

    for index in range(0, len(classified.index)):
        classified.loc[index, answerColumn], \
            classified.loc[index, confidenceColumn] = \
            formatAnswerClassificationType(str(classified[answerColumn][index]))
        progress.advance(task, advance = 1)

    progress.refresh()

# ------------------------------------------------------------------------------
# Persist transformed data to disk.
# ------------------------------------------------------------------------------

writeCSV(classified, outputFileClassificationTypeFormatted)

log("Logging incorrect classified Synonyms...")

# For logging purposes the gold standard is read and wrong classifications
# are placed in the logging file. This is useful when it comes to prompt 
# optimization.
gold        = readCSV(inputFileClassificationType)
labels      = gold[gold[classColumn] == labelClass].copy().reset_index(drop = True)
count       = 0

for index, row in classified.iterrows():
    # It should log, when:
    # - The answer could not be formatted e.g. is undefined.
    # - No type in the gold dataset means "expert", therefore having an empty
    #       type column and an answer that is not "expert" means wrong
    #       classification.
    # - If the type is direct, it just means that the source of the synonym is
    #       found in the class, not in the axioms, and therefore the type in 
    #       gold dataset is "expert". If the answer is not "expert" means wrong
    #       classification. 
    # - If the answer or the gold class is "layperson" but the other is not.
    # This way all "layperson" and "expert" terms classified the wrong way are
    # being logged.
    #
    if (str(row[answerColumn]).lower() == undefinedSynonymType.lower() or
        (str(row[answerColumn]).lower() != expertSynonymType and 
            row[typeColumn] == "") or
        (str(row[answerColumn]).lower() != expertSynonymType and 
            row[typeColumn] == directSynonymType) or
        ((str(row[answerColumn]).lower() != row[typeColumn] and 
            row[typeColumn] == laypersonSynonymType) and 
            (str(row[answerColumn]).lower() == laypersonSynonymType or 
                row[typeColumn] == laypersonSynonymType))):
        log(f"Label: " \
            f"{applyFormat(getElements(labels, row[hpoidColumn], labelClass))}" \
            f", Synonym: {quote(row[contentColumn])}, Correct: " \
            f"{quote(row[classColumn])}, Classified: {quote(row[answerColumn])}", 
            cmdline = False)
        count = count + 1

log(f"Incorrect Classifications: {count}")
log(f"Correct Classifications:   {len(classified.index) - count}")
log("Logging competed.")






# For time tracking.
minutes         = int((time.time() - start_time) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Formatting completed [Minutes: {minutes}]")