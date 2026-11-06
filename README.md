# synclass

Classifies the semantic class of the synonyms of [Human Phenotype Ontology](https://hpo.jax.org/) (HPO) concepts: whether a synonym means exactly the same as the label of its concept (`exact`) or only something related (`related`: broader, narrower, an abbreviation, another chemical form, ...). Local LLMs classify every synonym; the classifications are compared with the synonym classes HPO itself annotates, and with how similar a synonym is to the label of its concept.

## Features

- **Experiments in the configuration** — every experiment of `synclass.experiments` in [`config/config.yaml`](config/config.yaml) sets which information about a concept the prompts contain (definition, comment, parents and children), whether they contain few-shot examples, whether a synonym is classified in one prompt or in three chain-of-thoughts prompts, and whether the synonyms of the development set or of the evaluation set (all others) are classified.
- **Several models per experiment** — all models of `llm.model_id` classify the synonyms one after the other, in one Slurm job per experiment.
- **Resumable classification** — the [`model`](https://github.com/Goldkugel/model) package stores every prompt and answer, so an interrupted job resumes where it stopped and answers are never generated twice.
- **Distance baseline** — the cosine distance of every synonym to the label of its concept (SapBERT, via the [`embedder`](https://github.com/Goldkugel/embedder) package) shows how well wording alone separates exact from related synonyms.
- **Evaluation** — precision, recall and F1 per model and synonym class, a summary with bootstrap intervals against a majority and a distance baseline, confusion matrices, ROC curves, confidence calibration and distance plots.
- **Outputs made only when needed** — the transformed HPO, the development and evaluation set and the distances are made again only when their input is newer or their settings changed (recorded in `<output>.settings.json`).

## Repository Structure

```
synclass/
├── config/
│   ├── config.yaml          # Configuration: synclass, adapter, logger, llm and embedder sections
│   └── README.md
├── data/
│   ├── input/hpo/           # hp.owl
│   ├── logs/                # Log files, prompt logs and run metadata
│   ├── output/
│   │   ├── transformed/     # The HPO, development and evaluation set in the adapter's format
│   │   ├── embedded/        # Distance of every synonym to its label (development and evaluation set), embedding cache
│   │   ├── class/           # Raw classifications (prompts and answers)
│   │   ├── formatted/       # Formatted classifications, one file per model
│   │   ├── merged/          # Classifications of all models of an experiment
│   │   └── evaluated/       # Evaluation plots and summary
│   ├── slurmlogs/           # Output of the Slurm jobs
│   └── tmp/                 # Answer stores of the model package (not tracked)
├── prompts/                 # Prompt templates of the classification
├── src/
│   ├── SynClassClassification.py  # Preparation run and classification
│   ├── SynClassDevelopmentSet.py  # Draws the development and evaluation set
│   ├── SynClassMerge.py     # Merges the classifications of all models
│   ├── SynClassEval.py      # Evaluates the classifications
│   ├── SynClassUtils.py     # Helpers shared by the scripts
│   ├── SynClassConfig.py    # Schema of the synclass section
│   ├── config.py            # Configuration, experiment, file names, few-shot examples
│   └── README.md
├── start.sh                 # Preparation run, then one classification job per experiment
├── classify.sh              # Slurm job: classifies one experiment
├── finalize.sh              # Merges and evaluates every experiment
├── cleanUp.sh               # Deletes logs, classifications and evaluations
├── pyproject.toml
└── LICENSE
```

## Requirements & Installation

Python 3.10 or newer, and GPUs for the classification (the `model` package runs the LLMs with [vLLM](https://github.com/vllm-project/vllm)). The dependencies, including the sibling packages [`logger`](https://github.com/Goldkugel/logger), [`adapter`](https://github.com/Goldkugel/adapter), [`embedder`](https://github.com/Goldkugel/embedder) and [`model`](https://github.com/Goldkugel/model) at pinned versions, are declared in [`pyproject.toml`](pyproject.toml):

```bash
pip install -e .
```

Put the HPO release to classify at `data/input/hpo/hp.owl` (see [`data/input/README.md`](data/input/README.md)).

## Usage

All scripts are run from the repository root.

1. **Classify:** `./start.sh` runs the preparation (transforms the HPO, draws the development and evaluation set and computes the distances, each only if needed) and starts one Slurm job (`classify.sh`) per experiment, in which every model classifies the synonyms and its answers are formatted.
2. **Evaluate:** when the jobs are done, `./finalize.sh` merges the classifications of all models per experiment and evaluates them.

The Python scripts can also be run by hand:

| Command | What it does |
|---|---|
| `python3 ./src/SynClassClassification.py` | Preparation run: transforms the HPO, draws the development and evaluation set and embeds both, each only if needed |
| `python3 ./src/SynClassClassification.py <index>` | Classifies the synonyms of experiment `synclass.experiments[<index>]` (counted from 0) with every model of `llm.model_id`, one after the other |
| `python3 ./src/SynClassClassification.py <index> <model ID>` | The same with only this model |
| `python3 ./src/SynClassDevelopmentSet.py` | Draws the development and evaluation set again (e.g. after changing `synclass.development_set.seed`) |
| `python3 ./src/SynClassMerge.py <index>` | Merges the formatted classifications of the models of `llm.model_id` for the experiment |
| `python3 ./src/SynClassEval.py <index>` | Evaluates the merged classifications of the experiment |
| `python3 ./src/config.py` | Prints the number of experiments |

`./cleanUp.sh` deletes the logs, the classifications, the evaluations and the answer stores; the transformed HPO and the distances are kept.

## Configuration

[`config/config.yaml`](config/config.yaml) has one section per part: `synclass` (the experiments and synclass's own settings, see [`config/README.md`](config/README.md)), `adapter` (reading `hp.owl`), `logger`, `llm` (the models and the prompts) and `embedder` (the embedding model of the distances). Paths are relative to the repository root.

After the first preparation run, set `synclass.cosine_distance_mean` and `_std` to the mean and standard deviation it logs for the development set, and `synclass.distance_thresholds` to the threshold `SynClassEval.py` logs ("Distance baseline").

## Pipeline

| Step | Script | Output |
|---|---|---|
| Transformation | `SynClassClassification.py` | [`data/output/transformed/`](data/output/transformed/README.md): the HPO, without synonyms listed both as exact and related and synonyms equal to their label |
| Development and evaluation set | `SynClassClassification.py`, `SynClassDevelopmentSet.py` | [`data/output/transformed/`](data/output/transformed/README.md): 500 exact and 500 related synonyms at random, with their concepts, and the HPO without them |
| Distances | `SynClassClassification.py` | [`data/output/embedded/`](data/output/embedded/README.md): the cosine distance of every synonym to its label, for the development set (its mean and standard deviation are logged for `synclass.cosine_distance_mean` and `_std`) and for the evaluation set |
| Classification | `SynClassClassification.py <index>` | [`data/output/class/`](data/output/class/README.md): prompts and answers per model |
| Formatting | `SynClassClassification.py <index>` | [`data/output/formatted/`](data/output/formatted/README.md): the classification per model |
| Merging | `SynClassMerge.py <index>` | [`data/output/merged/`](data/output/merged/README.md): all models of an experiment |
| Evaluation | `SynClassEval.py <index>` | [`data/output/evaluated/`](data/output/evaluated/README.md): plots and summary |

The prompts are described in [`prompts/README.md`](prompts/README.md), the logs in [`data/logs/README.md`](data/logs/README.md).

## License

GNU General Public License v3.0, see [`LICENSE`](LICENSE).
