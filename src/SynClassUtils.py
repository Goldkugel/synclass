"""
Helpers shared by the scripts of synclass: the transformed HPO in the format of
the adapter package, the settings records that decide whether an output is
outdated, reading and writing CSV files, parsing the answers of the LLMs,
the labels of the concepts, the data summary and the development and
evaluation set.
"""

import pandas       as pd
import numpy as np
import ast
import json
import sys
import os
import textwrap

from rich.table     import Table
from rich.console   import Console

sys.dont_write_bytecode = True

from config         import * 

from logger         import Logger, writeJSON

from adapter                    import BaseAdapterConfig, semanticClass
from adapter                    import synonymClass as synonymAttribute
from adapter.BaseAdapterUtils   import additionalSeparator, isFile
from adapter.BaseAdapterUtils   import writeHugeCSV as writeHugeCSVAtomically

# The transformed HPO is in the adapter format: one row per (id, attribute,
# value) with a dict of additional information, e.g. {"semantic_class":
# "exact", "source_type": "layperson"}. Column names from adapter.hpo.
hpoAdapterConfig = BaseAdapterConfig.model_validate(configurationFile["adapter"]["hpo"])
idColumn         = hpoAdapterConfig.id_column
attributeColumn  = hpoAdapterConfig.attribute_column
valueColumn      = hpoAdapterConfig.value_column
additionalColumn = hpoAdapterConfig.additional_column

def additionalValues(additional, key : str) -> list:
    """
    The values of a key of the additional information of a row; a synonym
    listed several times for a concept has several, e.g. ["exact", "related"]
    (see the adapter's mergeAdditional()). [""] if the key is missing.
    """
    return str(additional.get(key, "")).split(additionalSeparator)

def synonymClassOf(additional) -> str:
    """
    The synonym class of a synonym (exact, related, ...), its gold class: its
    first one.
    """
    return additionalValues(additional, semanticClass)[0]

# ------------------------------------------------------------------------------
# The settings an output was made with are stored next to it
# (<output>.settings.json), so that it is made again when they change, not
# only when one of its inputs is newer.
# ------------------------------------------------------------------------------

def isOutdated(output : str, inputs : list) -> bool:
    """
    Whether a file is missing or older than one of the files it is made from.
    """
    return not isFile(output) or any(
        os.path.getmtime(output) < os.path.getmtime(i) for i in inputs if isFile(i))

def settingsFile(output : str) -> str:
    """
    The file holding the settings an output was made with.
    """
    return output + ".settings.json"

def changedSettings(output : str, settings : dict) -> list:
    """
    The settings (top-level keys) that differ from those the output was made
    with; all of them if they were not recorded.
    """
    try:
        with open(settingsFile(output), "r", encoding = "utf-8") as f:
            recorded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(settings.keys())

    # Through JSON, as the recorded settings were, e.g. tuples become lists.
    current = json.loads(json.dumps(settings))
    return [key for key in current if recorded.get(key) != current[key]]

def writeSettings(output : str, settings : dict) -> None:
    """
    Records the settings an output was made with, next to it (atomically).
    """
    writeJSON(settings, settingsFile(output))

def outdated(output : str, inputs : list, settings : dict, log : bool = True) -> bool:
    """
    Whether an output is missing, older than one of its inputs, or made with
    other settings; logs why, unless log is False.
    """
    if isOutdated(output, inputs):
        if log: Logger().log(f"{quote(os.path.basename(output))} is missing or older than " \
            f"{applyFormat([os.path.basename(i) for i in inputs])}.")
        return True

    changed = changedSettings(output, settings)
    if len(changed) > 0:
        if log: Logger().log(f"{quote(os.path.basename(output))} was made with other or unrecorded " \
            f"settings: {applyFormat(changed)}.")
        return True

    return False

def readTransformed(file : str) -> pd.DataFrame:
    """
    Read a transformed HPO file, with the additional information as dicts.
    """
    ret = readCSV(file)
    # Only few distinct dicts, so each is parsed once.
    parsed = {entry : ast.literal_eval(entry) for entry in ret[additionalColumn].astype(str).unique()}
    ret[additionalColumn] = ret[additionalColumn].astype(str).map(parsed)
    return ret

def synonymRows(data : pd.DataFrame) -> pd.Series:
    """
    Whether a row of the transformed HPO is a synonym synclass classifies (one
    of synonymClasses).
    """
    return (data[attributeColumn] == synonymAttribute) & \
        data[additionalColumn].map(synonymClassOf).isin(synonymClasses)

def synonymTable(data : pd.DataFrame) -> pd.DataFrame:
    """
    The synonyms of the transformed HPO that synclass classifies: their rows of
    the transformed HPO (id, attribute, value, additional), as in the
    classification and distance files.
    """
    return data[synonymRows(data)][
        [idColumn, attributeColumn, valueColumn, additionalColumn]].reset_index(drop = True)

def classesOf(data : pd.DataFrame) -> pd.Series:
    """The gold class of every row (its first, see synonymClassOf())."""
    return data[additionalColumn].map(synonymClassOf)

# A synonym of the transformed HPO is identified by its concept and its text.
synonymKeys = [idColumn, valueColumn]

def keysOf(data : pd.DataFrame) -> pd.Series:
    """The (concept, synonym) pair of every row, e.g. to compare files."""
    return pd.Series(list(zip(data[idColumn].astype(str), data[valueColumn].astype(str))),
        index = data.index)

# The distance columns: one per model of embedder.model_id and metric, the
# model named by synclass.embedding_model_names or its ID without organisation.
cosineDistance  = "cosine"
distanceMetrics = [cosineDistance]

embeddingModels = {
    synclassConfig.embedding_model_names.get(m, modelNameOf(m)) : m
    for m in configurationFile["embedder"]["model_id"]
}

def labelsByConcept(data : pd.DataFrame) -> dict:
    """
    The labels of every concept of a transformed HPO file, joined with ", "
    (a concept may have several), e.g. to name the concept of a synonym in
    the log; one pass over the data instead of a search per synonym.
    """
    labels = data[data[attributeColumn] == labelClass]
    return labels.groupby(idColumn, sort = False)[valueColumn].agg(
        lambda values: ", ".join(str(v) for v in values)).to_dict()


def exitIfFileNotExist(file: str) -> None:
    """
    Terminate the program if the specified file does not exist.
    """
    if not isFile(file):
        Logger().error(f"{quote(os.path.basename(file))} could not be found. Terminating.")
        sys.exit(1)


def readCSV(file: str = "") -> pd.DataFrame:
    """
    Read a CSV file from disk with logging.
    """
    Logger().printReadFileStart(file)
    ret = pd.read_csv(file, low_memory = False)
    Logger().printReadFileEnd(file)
    return ret


def writeHugeCSV(data: pd.DataFrame, file: str = "") -> None:
    """
    Write a DataFrame atomically as a comma-separated CSV file (see the
    adapter's writeHugeCSV()), creating its folder if needed.
    """
    folder = os.path.dirname(file)
    if folder:
        os.makedirs(folder, exist_ok = True)

    writeHugeCSVAtomically(data, file, ",")


def printDataSummary(data: pd.DataFrame = None) -> None:
    """
    Prints the absolute and relative frequencies of the attributes of a
    transformed HPO file, the synonyms per synonym class.
    """
    console = Console()

    table = Table(title="Data Summary Report")

    table.add_column("Description")
    table.add_column("Abs")
    table.add_column("Rel")

    # Synonyms are counted per synonym class.
    attributes   = data[attributeColumn].where(
        data[attributeColumn] != synonymAttribute,
        synonymAttribute + " (" + data[additionalColumn].map(synonymClassOf) + ")")
    value_counts = attributes.value_counts()

    for value, count in value_counts.items():
        relativeAmount = round(count * 100 / len(data.index))
        table.add_row(
            str(value),
            str(count).rjust(len(str(len(data.index)))),
            ("~" + str(relativeAmount) + "%").rjust(5)
        )

    console.print(table)


def parseConfidence(value) -> int | None:
    """
    The confidence of an answer as an integer of confidenceLevels, or None:
    an integer, or a number or text with an integer value (4, 4.0, "4");
    True, 4.5, "high" or a value outside the scale (e.g. 10 of an older
    scale) are no valid confidence.
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if not number.is_integer() or int(number) not in confidenceLevels:
        return None
    return int(number)

def parseClassification(answer) -> tuple[str, int, str]:
    """
    Classification, confidence and rationale of an answer parsed by
    model.toJSON(). Classification and confidence are undefinedSynonymClass
    and -1 unless the answer has one of synonymClasses and a confidence of
    confidenceLevels (see parseConfidence()); the rationale is "" if the
    answer has none.
    """
    classification, confidence = undefinedSynonymClass, -1
    rationale = ""

    if isinstance(answer, dict):
        if answer.get(rationaleColumn) is not None:
            rationale = str(answer[rationaleColumn]).strip()

        if classificationKey in answer and confidenceColumn in answer:
            parsedClassification = str(answer[classificationKey]).strip().lower()
            parsedConfidence     = parseConfidence(answer[confidenceColumn])

            if parsedClassification in synonymClasses and parsedConfidence is not None:
                classification, confidence = parsedClassification, parsedConfidence

    return classification, confidence, rationale

def quote(string : str) -> str:
    """
    The string in quotation marks, e.g. for the prompts and the log.
    """
    return quotationCharacter + string + quotationCharacter

def shorten(string, width : int = 60) -> str:
    """
    A text cut at a word to at most width characters, with " ..." where it
    is cut, so that a log row stays short.
    """
    return textwrap.shorten(str(string), width = width, placeholder = " ...")

def applyFormat(l : list[str] = []) -> str:
    """
    A list of strings as one text, each quoted and separated by commas, e.g.
    '"a", "b"'; a single string is only quoted, and an empty list gives
    "None".
    """
    string = "None"
    if len(l) > 0:
        if isinstance(l, str):
            string = quote(l)
        else:
            string = quote((quotationCharacter + ", " + 
                quotationCharacter).join(l))
    return string


# ------------------------------------------------------------------------------
# The development set (hpo.development.csv, classified by runs with "test")
# and the evaluation set (hpo.evaluation.csv, classified by the other runs).
# ------------------------------------------------------------------------------

# The settings both sets depend on (see outdated()).
developmentSetSettings = {
    "development_set" : synclassConfig.development_set.model_dump(),
    "gold_classes"    : synonymClasses,
    "format"          : [idColumn, attributeColumn, valueColumn, additionalColumn]
}

def developmentSetOutdated(log : bool = True) -> bool:
    """
    Whether the development or the evaluation set is missing, older than
    hpo.csv or drawn with other settings; the reason is logged if `log`.
    """
    return any([outdated(output, [outputFileTransformedFull], developmentSetSettings, log)
        for output in [outputFileTransformedDevelopment, outputFileTransformedEvaluation]])

def logSynonymCounts(name : str, data : pd.DataFrame) -> None:
    """
    Logs how many synonyms of every gold class a transformed HPO file holds,
    those synclass classifies (synonymTable()).
    """
    l = Logger()
    total = classesOf(synonymTable(data)).value_counts()
    l.log(f"{name}: {total.sum()} synonyms: " + ", ".join(
        f"{c} {n}" for c, n in total.items()) + ".")

def writeDevelopmentSet() -> pd.DataFrame:
    """
    Draws synclass.development_set.size_per_class synonyms of every gold class
    (exact, related) at random from hpo.csv and writes them with the label,
    definition, comment, parent and child relations of their concepts (and the
    labels of those parents and children) to hpo.development.csv. Writes
    hpo.csv without these synonyms to hpo.evaluation.csv, so that development
    synonyms are never evaluated. Records the settings next to both.

    :return: The development set.
    """
    l = Logger()
    settings = synclassConfig.development_set

    exitIfFileNotExist(outputFileTransformedFull)
    data    = readTransformed(outputFileTransformedFull)
    classes = data[additionalColumn].map(synonymClassOf)
    synonyms = synonymRows(data)

    selected = []
    for synonymClass in synonymClasses:
        candidates = data[synonyms & (classes == synonymClass)]
        if len(candidates.index) < settings.size_per_class:
            l.error(f"Only {len(candidates.index)} synonyms of gold class " \
                f"{quote(synonymClass)}, fewer than synclass.development_set." \
                f"size_per_class ({settings.size_per_class}).")
            sys.exit(1)
        selected.append(candidates.sample(n = settings.size_per_class,
            random_state = settings.seed).index)
        l.log(f"Drawing {settings.size_per_class} of {len(candidates.index)} " \
            f"synonyms of gold class {quote(synonymClass)} (seed {settings.seed}).")
    selectedRows = data.index.isin(np.concatenate(selected))

    # The information of their concepts, and the labels of their parents and
    # children (a child row reads "id has child value").
    concepts  = set(data.loc[selectedRows, idColumn])
    attribute = data[attributeColumn]
    conceptRows  = data[idColumn].isin(concepts) & attribute.isin(
        [labelClass, definitionClass, commentClass])
    relationRows = (attribute == childrenClass) & (
        data[idColumn].isin(concepts) | data[valueColumn].isin(concepts))
    related   = (set(data.loc[relationRows, idColumn]) |
        set(data.loc[relationRows, valueColumn])) - concepts
    relatedLabelRows = data[idColumn].isin(related) & (attribute == labelClass)

    result = data[selectedRows | conceptRows | relationRows | relatedLabelRows].reset_index(drop = True)
    l.log(f"Development set: {selectedRows.sum()} synonyms of {len(concepts)} concepts, " \
        f"{len(result.index)} rows.")
    l.log(f"Development set: {relationRows.sum()} relations, {len(related)} labels " \
        f"of parents and children.")
    printDataSummary(result)
    logSynonymCounts("Development set", result)

    # A drawn synonym listed several times is removed with all its rows.
    drawn       = set(zip(data.loc[selectedRows, idColumn], data.loc[selectedRows, valueColumn]))
    drawnRows   = synonyms & pd.Series(list(zip(data[idColumn], data[valueColumn])),
        index = data.index).isin(drawn)
    evaluation  = data[~drawnRows].reset_index(drop = True)
    l.log(f"Evaluation set: {len(evaluation.index)} of the {len(data.index)} rows of " \
        f"{quote(os.path.basename(outputFileTransformedFull))}.")
    l.log(f"Left out: {drawnRows.sum()} rows of the drawn synonyms.")
    logSynonymCounts("Evaluation set", evaluation)

    writeHugeCSV(result, outputFileTransformedDevelopment)
    writeSettings(outputFileTransformedDevelopment, developmentSetSettings)
    writeHugeCSV(evaluation, outputFileTransformedEvaluation)
    writeSettings(outputFileTransformedEvaluation, developmentSetSettings)
    return result
