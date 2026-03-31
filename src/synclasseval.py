import numpy                as np
import matplotlib.pyplot    as plt
import sys
import time
import math

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from model      import *
from utils      import *
from config     import *

from collections import Counter

printHeader("Evaluating the Results of Synonym Classification")
startTime = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassEvaluation)

classifiedComplete = readCSV(inputFileClassEvaluation)

ontologies = [
    str(s).split(":", 1)[0] for s in classifiedComplete[hpoidColumn].to_list()
]

counts = Counter(ontologies)

log(f"{len(counts.keys())} ontologies found.")

for key in counts.keys():
    log(f"Found {counts[key]} entries in HPO for ontology '{key}'.")

systems = list(set(classifiedComplete[systemColumn].tolist()))

string = "', '".join(systems)
log(f"Found Systems: {quote(string)}")
log(f"Classified Synonyms: {len(classifiedComplete.index)} " \
    f"(~{int(len(classifiedComplete.index) / len(systems))} per system)")
classifiedComplete = classifiedComplete[classifiedComplete[systemColumn] != ""]

# Change Datatype to String and lower case everything. 
classifiedComplete[classColumn]   = classifiedComplete[classColumn ].str.lower()
classifiedComplete[answerColumn]  = classifiedComplete[answerColumn].str.lower()
classifiedComplete[typeColumn]    = classifiedComplete[typeColumn  ].str.lower()

# Synonyms without a source type are set to "expert".
classifiedComplete[typeColumn]    = classifiedComplete[typeColumn].replace(
    np.nan, expertSynonymType.lower()
)

colors = plt.cm.tab10(range(len(systems) + 1))






for embeddingModel in embeddingModels.keys():
    for similarityMetric in similarityMetrics:
        column = similarityColumnPrefix.format(embeddingModel, similarityMetric)
        if column in classifiedComplete.columns:
            log(f"Plotting {similarityMetric} for Model {embeddingModel}...")

            x1 = []
            y1 = []
            x2 = []
            y2 = []
            for i in np.linspace(similarityEvaluationLowerBound, 
                                 similarityEvaluationUperBound, 
                                 similarityEvaluationParts):
                x1.append(i)
                x2.append(i)

                TP = sum((
                    classifiedComplete[column] < i
                        ) & (
                    classifiedComplete[classColumn] == relatedSynonymClass
                ))
                FP = sum((
                    classifiedComplete[column] < i
                        ) & (
                    classifiedComplete[classColumn] != relatedSynonymClass
                ))

                if TP + FP > 0:
                    y1.append((2 * TP) / (2 * TP + FP))
                else:
                    y1.append(np.nan)

                TP = sum((
                    classifiedComplete[column] >= i
                        ) & (
                    classifiedComplete[classColumn] == exactSynonymClass
                ))
                FP = sum((
                    classifiedComplete[column] >= i
                        ) & (
                    classifiedComplete[classColumn] != exactSynonymClass
                ))

                if TP + FP > 0:
                    y2.append((2 * TP) / (2 * TP + FP))
                else:
                    y2.append(np.nan)

            # Create the bar plot
            plt.figure()
            plt.plot(x1, y1, label = "F1 Score (Related Threshold)", 
                color = "red")
            plt.plot(x2, y2, label = "F1 Score (Exact Threshold)",   
                color = "blue")
            
            x1 = np.array(x1)
            y1 = np.array(y1)
            x2 = np.array(x2)
            y2 = np.array(y2)

            x1 = x1[y1 < 1]
            y1 = y1[y1 < 1]

            x2 = x2[y2 < 1]
            y2 = y2[y2 < 1]

            roundFactor = float(math.pow(10, len(str(similarityEvaluationParts))))

            if not np.all(np.isnan(y1)):
                max_index = np.nanargmax(y1)
                max_x = math.ceil(roundFactor * x1[max_index]) / roundFactor
                max_y = math.ceil(roundFactor * y1[max_index]) / roundFactor

                c = sum(classifiedComplete[column] < max_x)
                p = int(math.floor(roundFactor * c / len(classifiedComplete.index)))

                plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "red", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                log(f"Suggested Threshold for Class {quote(relatedSynonymClass)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)}" )

            if not np.all(np.isnan(y2)):
                max_index = np.nanargmax(y2)
                max_x = math.ceil(roundFactor * x2[max_index]) / roundFactor
                max_y = math.ceil(roundFactor * y2[max_index]) / roundFactor

                c = sum(classifiedComplete[column] >= max_x)
                p = int(math.floor(roundFactor * c / len(classifiedComplete.index)))

                plt.scatter(max_x, max_y, s = 200, marker = 'x', color = "blue", zorder = 100, label = "(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")
                log(f"Suggested Threshold for Class {quote(exactSynonymClass)}, Metric {quote(similarityMetric)}, and Model {quote(embeddingModel)}: {str(max_x)}" )

            plt.xlabel("Threshold")
            plt.ylabel("F1 Score")
            plt.title(f"Normalized {similarityMetric.capitalize()}-Similarity for Model {quote(embeddingModel)}")
            plt.grid(axis = "both")

            plt.yticks(np.linspace(0, 1.0, 11))
            plt.xticks(np.linspace(similarityEvaluationLowerBound, similarityEvaluationUperBound, 13))

            plt.legend(loc = "lower left", ncol = 1)

            outputFileClassEmbeddingEval              = os.path.join(
                dataDir,        
                outputFolderName,
                outputFolderNameClassEmbedding,
                outputFileNameClassEmbeddingEvaluation.format(column)
            )

            plt.savefig(outputFileClassEmbeddingEval, dpi = 300, bbox_inches = "tight")







classified = classifiedComplete.copy().drop([answerColumn, systemColumn], axis = 1).drop_duplicates().reset_index(drop = True)
classes = Counter(classified[classColumn])

# Prepare data for plotting
labels = list(classes.keys())
values = list(classes.values())

# Create the bar plot
plt.figure()
bars = plt.bar(labels, values, color = colors)

# Add counts on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha = 'center',
        va = 'bottom'
    )

plt.xlabel("Semantic Class")
plt.ylabel("Count")
plt.title("Count of Semantic Classes")
plt.grid(axis = "y")
plt.show()
plt.savefig(outputFileClassGoldCounts, dpi = 300, bbox_inches = "tight")







classifiedComplete = classifiedComplete[classifiedComplete[classColumn].isin(synonymClasses)].copy().reset_index(drop = True)

# Count occurrences per system and classification
classified = classifiedComplete
counts = classifiedComplete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)
# Create plot
ax = counts.plot(kind = "bar", figsize = (3 * len(systems), 4), width = 0.8)

# Add value labels above bars
for container in ax.containers:
    ax.bar_label(container, padding = 3)

plt.xlabel("Classified Semantic Class")
plt.ylabel("Count")
plt.title("Count of Classified Semantic Class")
plt.xticks(rotation = 0)
plt.grid(axis = "y")
plt.show()
plt.savefig(outputFileClassAnswerCounts, dpi = 300, bbox_inches = "tight")




# TODO: insert related class as well.
def ontologySubplot(data : pd.DataFame = None, startString : str = "", outputFile : str = "", title : str = "") -> None:
    classified = data[data[hpoidColumn].str.startswith(startString, na=False)].reset_index(drop = True).copy()

    result = {}

    for system in systems:
        systemData = classified[classified[systemColumn] == system]

        systemResults = {}

        if systemData is not None and len(systemData.index) > 0:
            for synonymClasse in synonymClasses:
                
                systemClassResults = {
                    precisionLabel  : 0,
                    recallLabel     : 0,
                    f1ScoreLabel    : 0
                }

                if len(systemData[systemData[answerColumn] == 
                    synonymClasse].index) > 0:

                    systemClassResults[precisionLabel] = \
                        len(systemData[
                                (systemData[classColumn] == synonymClasse
                                    ) & (
                                systemData[answerColumn] == synonymClasse)
                            ].index
                        ) / (
                            len(systemData[systemData[answerColumn] == 
                                synonymClasse].index) 
                        )
                    
                    systemClassResults[recallLabel] = \
                        len(systemData[
                                (systemData[classColumn] == synonymClasse
                                    ) & (
                                systemData[answerColumn] == synonymClasse)
                            ].index
                        ) / (
                            len(systemData[systemData[classColumn] == 
                                synonymClasse].index) 
                        )
                        
                    if systemClassResults[recallLabel] > 0 or systemClassResults[precisionLabel] > 0:
                        systemClassResults[f1ScoreLabel] = \
                            2 * systemClassResults[precisionLabel] * \
                            systemClassResults[recallLabel] / (
                            systemClassResults[recallLabel] + 
                            systemClassResults[precisionLabel])
                    else:
                        systemClassResults[f1ScoreLabel] = 0
                else:
                    systemClassResults[f1ScoreLabel]    = 0
                    systemClassResults[recallLabel]     = 0
                    systemClassResults[precisionLabel]  = 0

                systemResults[synonymClasse] = systemClassResults

        result[system] = systemResults

    metrics = [f1ScoreLabel, recallLabel, precisionLabel]
    systems = list(result.keys())
    classes = list(next(iter(result.values())).keys())

    x = np.arange(len(systems)) * 0.1 * len(systems)
    barWidth = 0.2

    fig, axes = plt.subplots(
        nrows = len(metrics),
        ncols = len(classes),
        figsize = (3 * len(metrics), 2 * len(classes)),
        sharey = True
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i])
        for i in range(0, len(systems))
    ]

    fig.legend(
        handles,
        systems,
        loc = "lower center",
        ncol = len(systems),
        frameon = False
    )

    for i, cls in enumerate(classes):
        for j, metric in enumerate(metrics):
            ax = axes[j, i]
            values = [result[system][cls][metric] for system in systems]

            for k, system in enumerate(systems):
                bars = ax.bar(
                    x[k],
                    values[k],
                    barWidth,
                    color = colors[k]
                )

                for bar in bars:
                    h = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.01,              # small vertical offset
                        f"{h:.2f}",
                        ha = "center",
                        va = "bottom",
                        fontsize = 7
                    )

            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels([""] * len(systems))

            if i == 0:
                ax.set_ylabel(metric.capitalize())
            if j == 0:
                ax.set_title(cls)

            ax.grid(axis="y")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect = [0, 0.05, 1, 1])
    plt.savefig(outputFile, dpi = 300, bbox_inches = "tight")

ontologySubplot(classifiedComplete, "",        outputFileClassRecallPrecisionF1,     "Per-Class and Per-Metric Comparison Across Systems for Full HPO Concepts")
ontologySubplot(classifiedComplete, "HP:",     outputFileClassEvaluationExactHPO,    "Exact Synonym Class Comparison Across Systems for HPO Only Concepts")
ontologySubplot(classifiedComplete, "UBERON:", outputFileClassEvaluationExactUBERON, "Exact Synonym Class Comparison Across Systems for UBERON Only Concepts")
ontologySubplot(classifiedComplete, "GO:",     outputFileClassEvaluationExactGO,     "Exact Synonym Class Comparison Across Systems for GO Only Concepts")
ontologySubplot(classifiedComplete, "CHEBI:",  outputFileClassEvaluationExactCHEBI,  "Exact Synonym Class Comparison Across Systems for CHEBI Only Concepts")