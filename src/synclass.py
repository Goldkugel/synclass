import sys
import logging
import time

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from prompts    import *
from utils      import *
from model      import *
from config     import *

logging.getLogger("vllm").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# Initialization.
# ------------------------------------------------------------------------------

printHeader(f"Classifying the Synonyms")

# To track time.
startTime = time.time()

# ------------------------------------------------------------------------------
# Load Human Phenotype Ontology (HPO) Data.
# ------------------------------------------------------------------------------

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClass)

# Load the dataset from a pickle file
gold    = readCSV(inputFileClass)

synonyms = gold[gold[classColumn].isin(synonymClasses)].copy().reset_index(
    drop = True)
hpoIDs   = getHPOIDs(synonyms)
parents  = {}
children = {}

with newProgress() as progress:
    
    task = newTask(progress, len(hpoIDs), "Get Parents and Children")

    for hpoID in hpoIDs:
        children[hpoID] = getChildLabels(gold, hpoID)
        parents[hpoID]  = getParentLabels(gold, hpoID)
        progress.update(task, advance = 1)
    
    progress.refresh()






# ------------------------------------------------------------------------------
# Classification of synonyms.
# ------------------------------------------------------------------------------

log(f"Set up the LLM ({modelName})...")
model = Model(model = modelID)
log(f"Set up of LLM complete.")

messages = []

synonyms = synonyms[synonyms[hpoidColumn].isin(hpoIDs)].copy().reset_index(
    drop = True)

with newProgress() as progress:

    task = newTask(progress, len(synonyms.index), "Set up first Prompt(s)")

    for index, row in synonyms.iterrows():
        hpoID = row[hpoidColumn]

        # For the Chain-Of-Thoughts approach there are several other prompts
        # following after this. The Few-Shot approach is incorporated directly
        # into the prompts.
        if chainOfThoughts:
            messages.append(semanticClassificationPrompt1(
                "".join(getElements(gold, hpoID, labelClass)),
                "".join(getElements(gold, hpoID, definitionClass)),
                "".join(getElements(gold, hpoID, commentClass)),
                parents[hpoID],
                children[hpoID]
            ))
        else:
            messages.append(semanticClassificationPrompt(
                "".join(getElements(gold, hpoID, labelClass)),
                "".join(getElements(gold, hpoID, definitionClass)),
                "".join(getElements(gold, hpoID, commentClass)),
                parents[hpoID],
                children[hpoID],
                row[contentColumn]
            ))

        progress.update(task, advance = 1)

    progress.refresh()

addedPrompts = model.addPrompt(userRole, messages)
log(f"{addedPrompts} prompts added. Start generating responses...")
model.generate()

# Here the other prompts for the Chain-Of-Thoughts approach are added.
# The Few-Shot approach is incorporated directl into the prompts.
if chainOfThoughts:
    messages = []
    with newProgress() as progress:

        task = newTask(progress, len(synonyms.index), "Set up second Prompt(s)")

        for index, row in synonyms.iterrows():
            hpoID = row[hpoidColumn]

            messages.append(semanticClassificationPrompt2(row[contentColumn]))
            progress.update(task, advance = 1)

        progress.refresh()

    addedPrompts = model.addPrompt(userRole, messages)
    log(f"{addedPrompts} prompts added. Start generating responses...")
    model.generate()

    addedPrompts = model.addPrompt(userRole, [semanticClassificationPrompt3()])
    log(f"{addedPrompts} prompts added. Start generating responses...")
    model.generate()

    addedPrompts = model.addPrompt(userRole, [semanticClassificationPrompt4(
        fewShot)])
    log(f"{addedPrompts} prompts added. Start generating responses...")
    model.generate()

log("Logging Prompts of Model...")
model.logPrompts()
log("Prompts of Model have been logged.")

# ------------------------------------------------------------------------------
# Clean data.
# ------------------------------------------------------------------------------

histories = model.getMessageHistories().copy()

synonyms[answerColumn] = [""] * len(synonyms.index)
for index, history in enumerate(histories):
    if (history is not None and 
        isinstance(history, list) and 
        messageTextElement in history[-1].keys() and 
        history[-1][messageTextElement] is not None):
        synonyms.loc[index, answerColumn] = \
            str(history[-1][messageTextElement]).strip()
        synonyms.loc[index, systemColumn] = modelName






# -----------------------------------------------------------------------------
# Persist transformed data to disk.
# -----------------------------------------------------------------------------

writeCSV(synonyms, outputFileClass)

# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Synonyms Classified [Minutes: {minutes}]")