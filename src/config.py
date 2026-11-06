"""
The configuration of synclass: reads config/config.yaml, picks the experiment
and the models of a run from the command line, and defines the column and
class names, the file names of every step and the Few-Shot examples.
Imported by every script with `from config import *`.

    python3 ./src/config.py   prints the number of experiments
"""

import os
import sys
import yaml

from SynClassConfig import ExperimentConfig, SynclassConfig

# The attributes and synonym classes of the transformed HPO, as the adapter
# writes them.
from adapter                  import labelClass, definitionClass, commentClass, childrenClass
from adapter                  import exactSynonymClass, relatedSynonymClass

sys.dont_write_bytecode = True

# =============================================================================
# Configuration File
# =============================================================================

# The settings of synclass ("synclass" section) and of the packages it uses
# (adapter, embedder, llm, logger).
configFile = "./config/config.yaml"

with open(configFile, "r", encoding = "utf-8") as f:
    configurationFile = yaml.safe_load(f)

synclassConfig = SynclassConfig.model_validate(configurationFile["synclass"])

# =============================================================================
# Experiment and models of a run, given on the command line:
#   (nothing)           the preparation run (SynClassClassification.py only);
#   <index>             the experiment synclass.experiments[<index>], from 0,
#                       with every model of llm.model_id;
#   <index> <model ID>  SynClassClassification.py with only this model.
# =============================================================================

experimentIndex = None
if len(sys.argv) > 1:
    experimentCount = len(synclassConfig.experiments)
    if not sys.argv[1].isdigit() or int(sys.argv[1]) >= experimentCount:
        sys.exit(f"The experiment '{sys.argv[1]}' does not exist: give its index in " \
            f"synclass.experiments, 0 to {experimentCount - 1}.")
    experimentIndex = int(sys.argv[1])

# The preparation run classifies nothing, so it has no experiment.
experiment = (ExperimentConfig() if experimentIndex is None
    else synclassConfig.experiments[experimentIndex])

reduceToTestIDs         = experiment.test
fewShot                 = experiment.few_shot
chainOfThoughts         = experiment.chain_of_thoughts
addDefinition           = experiment.definition
addComment              = experiment.comment
addStructure            = experiment.structure

forceDrawAdditionalPlots = synclassConfig.force_draw_additional_plots

testStr                 = "Test" if reduceToTestIDs else "NoTest"
configStr               = "Def" if addDefinition else "NoDef"
configStr              += "_Com" if addComment else "_NoCom"
configStr              += "_Str" if addStructure else "_NoStr"
fewShotStr              = "FS" if fewShot else "NoFS"
chainOfThoughtsStr      = "COT" if chainOfThoughts else "NoCOT"

# =============================================================================
# Model Configuration
# =============================================================================

# None in the preparation run (empty), the one given after the experiment, or
# all of llm.model_id.
modelIDs = []
if experimentIndex is not None:
    modelIDs = sys.argv[2:3] or list(configurationFile["llm"]["model_id"])

def modelNameOf(modelID : str) -> str:
    """
    The name of a model in file names and the system column: its ID without
    the organisation, e.g. "medgemma-4b-it" for "google/medgemma-4b-it".
    """
    return modelID[modelID.index("/") + 1:] if "/" in modelID else modelID

# The distance columns, distance_<embedding model>_<metric>.
distanceColumnPrePrefix   = "distance_"
distanceColumnPrefix      = f"{distanceColumnPrePrefix}" + "{}_{}"

distanceEvaluationLowerBound = synclassConfig.distance_evaluation_lower_bound
distanceEvaluationUpperBound = synclassConfig.distance_evaluation_upper_bound
# The number of thresholds tried, one every distance_evaluation_step of the
# configured range (rounded, as e.g. 8 / 0.001 is not exactly 8000).
distanceEvaluationParts      = round((distanceEvaluationUpperBound - distanceEvaluationLowerBound)
    / synclassConfig.distance_evaluation_step) + 1

# The bootstrap samples of the intervals in the evaluation summary.
bootstrapSamples             = synclassConfig.bootstrap_samples

# The thresholds of the distance models of the evaluation.
distanceThresholds           = synclassConfig.distance_thresholds

# Mean and standard deviation of the cosine distance with which the distances
# are normalised (None: computed from the data).
cosineDistanceMean           = synclassConfig.cosine_distance_mean
cosineDistanceStd            = synclassConfig.cosine_distance_std

# =============================================================================
# Formatting of the prompts and the log
# =============================================================================

quotationCharacter = "\""

# =============================================================================
# Result columns of the classification files, after the columns of the
# transformed HPO (id, attribute, value, additional)
# =============================================================================
systemColumn                = "system"
answerColumn                = "answer"
confidenceColumn            = "confidence"
# The short explanation of a classification in the LLM's answer.
rationaleColumn             = "rationale"

# =============================================================================
# The answer key, and the synonym classes synclass classifies
# =============================================================================

# The key of the classification in the LLM's JSON answer.
classificationKey               = "classification"

# The synonym classes synclass classifies, and the classes the LLM classifies
# them into (see prompts/); broad and narrow synonyms name a broader or
# narrower concept and are left out.
synonymClasses = [exactSynonymClass, relatedSynonymClass]

# An answer without a valid classification.
undefinedSynonymClass           = "undefined"

# The confidence scale of the answers (see prompts/), 1 to 5.
confidenceLevels = range(1, 6)

# The metrics of the precision, recall and F1 plot.
precisionLabel  = "precision"
recallLabel     = "recall"
f1ScoreLabel    = "f1"

# =============================================================================
# Folder structure
# =============================================================================

csvFileFormat = "csv"
plotFileFormat = "png"

dataDir = synclassConfig.data_folder

outputFolderName                    = "output"

# The folders of the formatting, merging and evaluation under data/output/;
# the raw classifications are in outputFolderNameClass, the distances in
# outputFolderNameEmbedded, the transformed HPO in adapter.hpo.output_folder.
outputFolderNameFormatted           = "formatted"
outputFolderNameMerged              = "merged"
outputFolderNameEvaluation          = "evaluated"


# The HPO file read by the adapter (adapter.hpo section).
inputFileTransformed        = os.path.join(
    configurationFile["adapter"]["hpo"]["input_folder"],
    configurationFile["adapter"]["hpo"]["input_files"][0]
)

# hp.owl transformed by SynClassClassification.py, where adapter.hpo.output_folder
# and output_file say: everything the classification and the evaluation need.
# The development and evaluation set are named after it in the same folder,
# e.g. hpo.development.csv and hpo.evaluation.csv.
outputFileTransformedFull = os.path.join(
    configurationFile["adapter"]["hpo"]["output_folder"],
    configurationFile["adapter"]["hpo"]["output_file"]
)
transformedStem, transformedExtension = os.path.splitext(outputFileTransformedFull)

# The development set (synclass.development_set): the synonyms drawn at random
# and the information of their concepts, classified by runs with "test".
outputFileTransformedDevelopment = f"{transformedStem}.development{transformedExtension}"

# The evaluation set: hpo.csv without the synonyms of the development set,
# classified by runs without "test".
outputFileTransformedEvaluation  = f"{transformedStem}.evaluation{transformedExtension}"

inputFileTask = outputFileTransformedDevelopment if reduceToTestIDs else outputFileTransformedEvaluation

# Distances of the synonyms to their label (the preparation run), in the
# columns of hpo.csv: of the development set, for the evaluation of test runs
# and to choose the distance threshold.
outputFolderNameEmbedded             = "embedded"
outputFileEmbeddedDevelopment        = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEmbedded,
    f"embed.development.{csvFileFormat}"
)

# Of the evaluation set, for the evaluation of the other runs.
outputFileEmbeddedEvaluation         = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEmbedded,
    f"embed.{csvFileFormat}"
)

inputFileDistances = outputFileEmbeddedDevelopment if reduceToTestIDs else outputFileEmbeddedEvaluation

# =============================================================================
# Files for Synonym Classification
# =============================================================================

# The short experiment name in every file name of a run, e.g.
# class_Test_NoDef_NoCom_NoStr_NoCOT_FS.
outputFolderNameClass    = "class"
filesPrefixClass         = f"{outputFolderNameClass}_{testStr}_{configStr}"
filesPrefixApproachClass = f"{filesPrefixClass}_{chainOfThoughtsStr}_{fewShotStr}"

# The synonyms the runs of the experiment classify.
inputFileClass                         = inputFileTask

outputFileNameClassFormattedPrefix     = f"{filesPrefixApproachClass}_formatted"

def outputFileClass(modelName : str) -> str:
    """
    The raw classification of a model: prompts, answers and the parsed
    classification.
    """
    return os.path.join(dataDir, outputFolderName, outputFolderNameClass,
        f"{filesPrefixApproachClass}_raw_{modelName}.{csvFileFormat}")

def outputFileClassFormatted(modelName : str) -> str:
    """
    The formatted classification of a model, merged by SynClassMerge.py.
    """
    return os.path.join(dataDir, outputFolderName, outputFolderNameFormatted,
        f"{outputFileNameClassFormattedPrefix}_{modelName}.{csvFileFormat}")

# Read by SynClassMerge.py.
inputFolderClassMerged                 = os.path.join(dataDir, outputFolderName, outputFolderNameFormatted)

outputFileNameClassMerged              = f"{filesPrefixApproachClass}_merged_classes.{csvFileFormat}"
outputFileClassMerged                  = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameMerged,
    outputFileNameClassMerged
)

inputFileClassEvaluation               = outputFileClassMerged

outputFileNameClassGoldCounts   = f"{filesPrefixApproachClass}_gold_counts.{plotFileFormat}"
outputFileClassGoldCounts       = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassGoldCounts
)

outputFileNameClassRecallPrecisionF1   = f"{filesPrefixApproachClass}_base_evaluation.{plotFileFormat}"
outputFileClassRecallPrecisionF1 = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassRecallPrecisionF1
)

outputFileNameClassAnswerCounts   = f"{filesPrefixApproachClass}_answer_counts.{plotFileFormat}"
outputFileClassAnswerCounts = os.path.join(
    dataDir,
    outputFolderName,
    outputFolderNameEvaluation,
    outputFileNameClassAnswerCounts
)

# The distance plots, one per distance column; named without the prompt
# settings, which do not change the distances.
outputFolderNameClassEmbedding = outputFolderNameEvaluation

embeddingPrefix = f"{outputFolderNameClass}_{testStr}"

outputFileNameClassEmbeddingEvaluation    = f"{embeddingPrefix}_embedding_evaluation_" + "{}" + f".{plotFileFormat}"
outputFileNameClassDistanceEvaluation   = f"{embeddingPrefix}_distance_evaluation_" + "{}" + f".{plotFileFormat}"

# The summary table of the models and baselines, and the confusion matrices,
# ROC curves and confidence calibration of the models.
def outputFileClassEvaluationOf(name : str, fileFormat : str = plotFileFormat) -> str:
    """The path of an evaluation output of the experiment, e.g. ..._summary.csv."""
    return os.path.join(dataDir, outputFolderName, outputFolderNameEvaluation,
        f"{filesPrefixApproachClass}_{name}.{fileFormat}")

outputFileClassSummary     = outputFileClassEvaluationOf("summary", csvFileFormat)
outputFileClassConfusion   = outputFileClassEvaluationOf("confusion_matrices")
outputFileClassROC         = outputFileClassEvaluationOf("roc")
outputFileClassCalibration = outputFileClassEvaluationOf("calibration")


# The Few-Shot examples, alternately exact and related, with their answer
# (classification, confidence from 1 to 5, rationale). Their synonyms are left
# out of the evaluation also in runs without examples, so that all runs are
# scored on the same synonyms.
fewShotExamples = [
    {
        "id"        : "HP:0008984",
        "label"     : "Neck muscle hypoplasia",
        "definition": "Underdevelopment of muscles of the neck.",
        "comment"   : "",
        "parents"   : ["Hypoplasia of the musculature", "Abnormal neck morphology"],
        "children"  : [],
        "synonym"   : "Underdevelopment of neck muscle",
        "class"     : exactSynonymClass,
        "confidence": "5",
        "rationale" : "'Underdevelopment' is the plain-language rendering of 'hypoplasia', and the synonym names the same structure. Both terms denote the same phenotype and can be used interchangeably."
    },
    {
        "id"        : "CHEBI:16375",
        "label"     : "D-cysteine",
        "definition": "An optically active form of cysteine having D-configuration.",
        "comment"   : "",
        "parents"   : ["cysteine", "D-alpha-amino acid"],
        "children"  : [],
        "synonym"   : "DCY",
        "class"     : relatedSynonymClass,
        "confidence": "4",
        "rationale" : "'DCY' is an abbreviation (a database code) of the compound, not a name of it; without context it is ambiguous. It refers to the concept but cannot replace the label."
    },
    {
        "id"        : "UBERON:0000401",
        "label"     : "mandibular ramus",
        "definition": "The upturned perpendicular extremity of the mandible.",
        "comment"   : "",
        "parents"   : ["zone of bone organ"],
        "children"  : [],
        "synonym"   : "ramus of mandible",
        "class"     : exactSynonymClass,
        "confidence": "5",
        "rationale" : "The synonym only rewords the label ('mandibular ramus' as 'ramus of mandible'), the anatomical structure is the same. Anatomists use both interchangeably."
    },
    {
        "id"        : "HP:0001250",
        "label"     : "Seizure",
        "definition": "A seizure is an intermittent abnormality of nervous system physiology characterized by a transient occurrence of signs and/or symptoms due to abnormal excessive or synchronous neuronal activity in the brain.",
        "comment"   : "A type of electrographic seizure has been proposed in neonates which does not have a clinical correlate, it is electrographic only. The term epilepsy is not used to describe recurrent febrile seizures. Epilepsy presumably reflects an abnormally reduced seizure threshold.",
        "parents"   : ["Abnormal nervous system physiology"],
        "children"  : ["Bilateral tonic-clonic seizure", "Status epilepticus", "Generalized-onset seizure", "Focal-onset seizure", "Symptomatic seizures", "Dialeptic seizure", "Reflex seizure", "Motor seizure", "Nocturnal seizures", "Neonatal seizure", "Infection-related seizure", "Non-motor seizure"],
        "synonym"   : "Epilepsy",
        "class"     : relatedSynonymClass,
        "confidence": "5",
        "rationale" : "Epilepsy is a disease of recurrent unprovoked seizures, whereas a seizure is a single event that also occurs without epilepsy. The terms are closely associated but not interchangeable."
    }
]


# The number of experiments, for the shell scripts (start.sh, finalize.sh).
if __name__ == "__main__":
    print(len(synclassConfig.experiments))
