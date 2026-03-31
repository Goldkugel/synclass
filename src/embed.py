import pandas as pd
import numpy as np
import sys
import time
import torch
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Prevent Python from generating .pyc files
sys.dont_write_bytecode = True

# Project-specific configuration and utility functions
from config import *
from utils  import *

printHeader("Embedding HPO Terms")

# To track time.
start_time = time.time()





data = readCSV(inputFileTask)

hpoIDs = getHPOIDs(data)

labels      = data[(data[classColumn] == labelClass) & (data[hpoidColumn].isin(hpoIDs))][contentColumn].tolist()
labelIDs    = data[(data[classColumn] == labelClass) & (data[hpoidColumn].isin(hpoIDs))][hpoidColumn].tolist()

synonyms    = data[(data[classColumn].isin(synonymClasses)) & (data[hpoidColumn].isin(hpoIDs))][contentColumn].tolist()
synonymIDs  = data[(data[classColumn].isin(synonymClasses)) & (data[hpoidColumn].isin(hpoIDs))][hpoidColumn].tolist()

for model in embeddingModels:

    log(f"Set up model {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model,   model_max_length = 64)
    model = AutoModel.from_pretrained(model)
    # Put model in evaluation mode
    model.eval()
    log("Model set up.")


    log("Tokenize Labels...")
    # Tokenize input
    inputs = tokenizer(
        labels,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    log("Labels tokenized.")
    

    log("Embedding Labels...")
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    log("Labels embedded.")

    log("Finalizing embeddings of Labels...")
    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    # Normalize embeddings (optional but common)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    resultLabels = {}
    # Print results
    for i, l, e in zip(ids, labels, embeddings):
        resultLabels[i] = e
    log("Finalized Label embeddings.")


    log("Tokenize Synonyms...")
    # Tokenize input
    inputs = tokenizer(
        synonyms,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    log("Synonyms tokenized.")

    log("Embedding Synonyms...")
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    log("Synonyms embedded.")

    log("Finalizing embeddings of Synoyms...")
    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    resultSynonyms = {}
    # Print results
    for i, s, e in zip(ids, synonyms, embeddings):
        resultSynonyms[str(i) + " " + str(s)] = e
    log("Finalized Synonym embeddings.")

    for similarityMetric in similarityMetrics:
        similarityColumn = similarityColumnPrefix + similarityMetric
        data[similarityColumn] = [np.nan] * len(data.index)

    with newProgress() as progress:

        task = newTask(progress, len(data.index), "Calculating Similarity")

        for index, row in data.iterrows():
            if row[classColumn] in synonymClasses and row[hpoidColumn] in hpoIDs:
                emb1 = resultSynonyms[str(row[hpoidColumn]) + " " + str(row[contentColumn])].unsqueeze(0)
                emb2 = resultLabels[row[hpoidColumn]].unsqueeze(0)

                emb1 = emb1[0]
                emb2 = emb2[0]

                for similarityMetric in similarityMetrics:
                    similarityColumn = similarityColumnPrefix + similarityMetric

                    if similarityMetric == cosineSimilarity:
                        data.loc[index, similarityColumn] = cosSim(emb1, emb2)

                    if similarityMetric == euclideanSimilarity:
                        data.loc[index, similarityColumn] = eucSim(emb1, emb2)

                    if similarityMetric == manhattanSimilarity:
                        data.loc[index, similarityColumn] = manSim(emb1, emb2)

                    if similarityMetric == angularSimilarity:
                        data.loc[index, similarityColumn] = angSim(emb1, emb2)

                    if similarityMetric == mahalanobisSimilarity:
                    data.loc[index, similarityColumn] = mahSim(emb1, emb2)

            progress.update(task, advance = 1)

        progress.refresh()

    for similarityMetric in similarityMetrics:
        similarityColumn = similarityColumnPrefix + similarityMetric
        data[similarityColumn] = (data[similarityColumn] - data[similarityColumn].mean()) / data[similarityColumn].std()

    m = "mean"
    s = "std"

    for similarityMetric in similarityMetrics:
        similarityColumn = similarityColumnPrefix + similarityMetric

        grouped = data.groupby(classColumn)[similarityColumn].agg([m, s])

        means = pd.Series(grouped[m].tolist()).dropna().tolist()
        stds = pd.Series(grouped[s].tolist()).dropna().tolist()

        log(similarityMetric + " Difference: [" + exactSynonymClass + " - " + 
            relatedSynonymClass + "]")
        log("Means: "               + str(means[0] - means[1]))
        log("Standard Deviation: " + str(stds[0] - stds[1]))
        log(similarityMetric + " Values: [" + exactSynonymClass + ", " + 
            relatedSynonymClass + "]")
        log("Means: "               + str(means))
        log("Standard Deviation: "  + str(stds))


        ssmd = math.fabs(means[0] - means[1]) / math.sqrt(math.fabs(math.pow(stds[0], 2) - math.pow(stds[1], 2)))

        log("Strictly standardized mean difference: " + str(ssmd))

writeCSV(data, inputFileTask)






minutes         = int((time.time() - start_time) // 60)

printHeader(f"Embedding of HPO Terms completed [Minutes: {minutes}]")