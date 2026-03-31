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
start_time = time.time()

# Only proceed if formatted input data exists
exitIfFileNotExist(inputFileClassificationEvaluation)

classified_complete = readCSV(inputFileClassificationEvaluation)

ontologies = [str(s).split(":", 1)[0] for s in classified_complete[hpoidColumn].to_list()]

counts = Counter(ontologies)

for key in counts.keys():
    if (counts[key] > 1000):
        log(f"Found {counts[key]} entries in HPO for ontology '{key}'.")

systemsName = list(set(classified_complete[systemColumn].tolist()))

string = "', '".join(systemsName)
log(f"Found Systems: '{string}'")
log(f"Classified Synonyms: {len(classified_complete.index)} (~{int(len(classified_complete.index) / len(systemsName))} per system)")
classified_complete   = classified_complete[classified_complete[systemColumn] != ""]

# Change Datatype to String. 
classified_complete[classColumn]    = classified_complete[classColumn ].str.lower()
classified_complete[answerColumn]   = classified_complete[answerColumn].str.lower()
classified_complete[typeColumn]     = classified_complete[typeColumn  ].str.lower()

classified_complete[typeColumn]     = classified_complete[typeColumn].replace(np.nan, expertSynonymType)



systems     = list(set(classified_complete[systemColumn].tolist()))
classificationClasses = list(set(classified_complete[classColumn]))

colors = plt.cm.tab10(range(len(systems) + 1))






for col in list(classified_complete.columns):
    if similarityColumnPrePrefix in col:
        parts = str(col).split("_")
        metricName = str(parts[-1]).capitalize() + " Similarity"
        modelName = "_".join(parts[1:len(parts)-1])
        log(f"Plotting {metricName} for Model {modelName}...")

        n = 1000
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in np.linspace(-3, 3, n):
            x1.append(i)
            x2.append(i)

            TP = sum((classified_complete[col] < i) & (classified_complete[classColumn] == relatedSynonymClass))
            FP = sum((classified_complete[col] < i) & (classified_complete[classColumn] == exactSynonymClass))
            if TP + FP > 0:
                y1.append((2 * TP) / (2 * TP + FP))
            else:
                y1.append(np.nan)

            TP = sum((classified_complete[col] >= i) & (classified_complete[classColumn] == exactSynonymClass))
            FP = sum((classified_complete[col] >= i) & (classified_complete[classColumn] == relatedSynonymClass))
            if TP + FP > 0:
                y2.append((2 * TP) / (2 * TP + FP))
            else:
                y2.append(np.nan)

        # Create the bar plot
        plt.figure()
        plt.plot(x1, y1, label = "F1 Score (Related Threshold)", color="red")
        plt.plot(x2, y2, label = "F1 Score (Exact Threshold)", color="green")
        
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)

        x1 = x1[y1 < 1]
        y1 = y1[y1 < 1]

        x2 = x2[y2 < 1]
        y2 = y2[y2 < 1]

        if not np.all(np.isnan(y1)):
            max_index = np.nanargmax(y1)
            max_x = math.ceil(100 * x1[max_index]) / 100
            max_y = math.ceil(100 * y1[max_index]) / 100

            c = sum(classified_complete[col] < max_x)
            p = int(math.floor(100.0 * c / len(classified_complete.index)))

            plt.scatter(max_x, max_y, s=200, marker='x', color="red", zorder=100, label="(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")

        if not np.all(np.isnan(y2)):
            max_index = np.nanargmax(y2)
            max_x = math.ceil(100 * x2[max_index]) / 100
            max_y = math.ceil(100 * y2[max_index]) / 100

            c = sum(classified_complete[col] >= max_x)
            p = int(math.floor(100.0 * c / len(classified_complete.index)))

            plt.scatter(max_x, max_y, s=200, marker='x', color="green", zorder=100, label="(" + str(max_x) + ", " + str(max_y) + ") (Count: " + str(c) + ", " + str(p) + "%)")

        plt.xlabel("Threshold")
        plt.ylabel("")
        plt.title(metricName + " for Model \"" + modelName + "\"")
        plt.grid(axis = "both")

        plt.yticks(np.linspace(0    , 1.0, 11))
        plt.xticks(np.linspace(-3.0 , 3.0, 13))

        plt.legend(loc="lower left", ncol=1)

        outputFileEmbeddingEval              = os.path.join(
            dataDir,        
            outputFolderName,
            outputFolderNameEmbedding,
            outputFileNameEmbeddingEvaluation.format(col)
        )

        plt.savefig(outputFileEmbeddingEval, dpi = 300, bbox_inches = "tight")

classified = classified_complete.copy().drop([answerColumn, systemColumn], axis=1).drop_duplicates().reset_index(drop = True)
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
        ha='center',
        va='bottom'
    )

plt.xlabel("Semantic Class")
plt.ylabel("Count")
plt.title("Count of Semantic Classes")
plt.grid(axis = "y")
plt.show()
plt.savefig(outputFileClassificationGoldCounts, dpi = 300, 
    bbox_inches = "tight")

classified_complete = classified_complete[classified_complete[classColumn].isin(evaluationClasses)].copy().reset_index(drop = True)







# Count occurrences per system and classification
classified = classified_complete
counts = classified_complete.groupby([answerColumn, systemColumn]).size().unstack(fill_value = 0)
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
plt.savefig(outputFileClassificationAnswerCounts, dpi = 300, 
    bbox_inches = "tight")






classified = classified_complete.copy()

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for evaluationClass in evaluationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                evaluationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            evaluationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            evaluationClass].index) 
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

            systemResults[evaluationClass] = systemClassResults

    result[system] = systemResults

metrics = [f1ScoreLabel, recallLabel, precisionLabel]
systems = list(result.keys())
classes = list(next(iter(result.values())).keys())

x = np.arange(len(systems)) * 0.1 * len(systems)
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=len(classes),
    figsize=(3 * len(metrics), 2 * len(classes)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for i, cls in enumerate(classes):
    for j, metric in enumerate(metrics):
        ax = axes[j, i]
        values = [result[system][cls][metric] for system in systems]

        for k, system in enumerate(systems):
            bars = ax.bar(
                x[k],
                values[k],
                bar_width,
                color=colors[k]
            )

            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,              # small vertical offset
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(systems))

        if i == 0:
            ax.set_ylabel(metric.capitalize())
        if j == 0:
            ax.set_title(cls)

        ax.grid(axis="y")

fig.suptitle("Per-Class and Per-Metric Comparison Across Systems for Full HPO Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationRecallPrecisionF1, dpi=300, bbox_inches="tight")






classes = [exactSynonymClass]

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classes),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classes)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for Full HPO Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExact, dpi=300, bbox_inches="tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("HP:", na=False)].reset_index(drop = True).copy()

systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for evaluationClass in evaluationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                evaluationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            evaluationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            evaluationClass].index) 
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

            systemResults[evaluationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for HPO Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactHPO, dpi=300, bbox_inches="tight")




classified = classified_complete[classified_complete[hpoidColumn].str.startswith("UBERON:", na=False)].reset_index(drop = True).copy()
systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for evaluationClass in evaluationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                evaluationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            evaluationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            evaluationClass].index) 
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

            systemResults[evaluationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for UBERON Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactUBERON, dpi=300, bbox_inches="tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("GO:", na=False)].reset_index(drop = True).copy()
systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for evaluationClass in evaluationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                evaluationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            evaluationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            evaluationClass].index) 
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

            systemResults[evaluationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for GO Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactGO, dpi=300, bbox_inches="tight")



classified = classified_complete[classified_complete[hpoidColumn].str.startswith("CHEBI:", na=False)].reset_index(drop = True).copy()
systems     = list(set(classified[systemColumn].tolist()))
classificationClasses = [exactSynonymClass]
colors = plt.cm.tab10(range(len(systems) + 1))

result = {}

for system in systems:
    systemData = classified[classified[systemColumn] == system]

    systemResults = {}

    if systemData is not None and len(systemData.index) > 0:
        for evaluationClass in evaluationClasses:
            
            systemClassResults = {
                precisionLabel  : 0,
                recallLabel     : 0,
                f1ScoreLabel    : 0
            }

            if len(systemData[systemData[answerColumn] == 
                evaluationClass].index) > 0:

                systemClassResults[precisionLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[answerColumn] == 
                            evaluationClass].index) 
                    )
                
                systemClassResults[recallLabel] = \
                    len(systemData[
                            (systemData[classColumn] == evaluationClass
                                ) & (
                            systemData[answerColumn] == evaluationClass)
                        ].index
                    ) / (
                        len(systemData[systemData[classColumn] == 
                            evaluationClass].index) 
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

            systemResults[evaluationClass] = systemClassResults

    result[system] = systemResults

x = np.arange(len(systems)) * 0.2
bar_width = 0.2

fig, axes = plt.subplots(
    nrows=len(classificationClasses),
    ncols=len(metrics),
    figsize=(3 * len(metrics), 2 * len(classificationClasses)),
    sharey=True
)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i])
    for i in range(0, len(systems))
]

fig.legend(
    handles,
    systems,
    loc="lower center",
    ncol=len(systems),
    frameon=False
)

colors = plt.cm.tab10(range(len(systems) + 1))

for j, metric in enumerate(metrics):
    ax = axes[j]
    values = [result[system][exactSynonymClass][metric] for system in systems]

    for k, system in enumerate(systems):
        bars = ax.bar(
            x[k],
            values[k],
            bar_width,
            color=colors[k]
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h - 0.15,              # small vertical offset
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(systems))

    ax.set_title(metric)

    ax.grid(axis="y")

fig.suptitle("Exact Synonym Class Comparison Across Systems for CHEBI Only Concepts", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(outputFileClassificationEvaluationExactCHEBI, dpi=300, bbox_inches="tight")



end_time = time.time()
elapsed_seconds = end_time - start_time
minutes = int(elapsed_seconds // 60)

# Print a formatted header indicating the end of this processing stage
printHeader(f"Data Evaluated [Minutes: {minutes}]")