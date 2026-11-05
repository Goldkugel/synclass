# Source

The scripts of synclass, run from the repository root (see the [README](../README.md)).

| File | Content |
|---|---|
| `SynClassClassification.py` | Without arguments the preparation run: transforms `hp.owl` with the `adapter` package into `hpo.csv`, draws the development set (`hpo.development.csv`) and the evaluation set (`hpo.evaluation.csv`) and embeds the synonyms and labels of both with the `embedder` package into `embed.development.csv` and `embed.csv`, each only if needed; the mean and standard deviation of the development set are logged for `synclass.cosine_distance_mean` and `_std`. With an experiment index: classifies the synonyms with every model of `llm.model_id` (or the one given) through the `model` package and formats the answers |
| `SynClassDevelopmentSet.py` | Draws the development set again (`writeDevelopmentSet()` in `SynClassUtils.py`): exact and related synonyms at random from `hpo.csv`, with the label, definition, comment, parent and child relations of their concepts and the labels of those parents and children, into `hpo.development.csv`, and `hpo.csv` without these synonyms into `hpo.evaluation.csv`. The preparation run draws them as well when one is missing or outdated |
| `SynClassMerge.py` | Merges the formatted classifications of the models of `llm.model_id` for an experiment into one file, and logs missing models and differing synonyms |
| `SynClassEval.py` | Evaluates the merged classifications of an experiment: counts, precision, recall and F1 per model and synonym class, and distance plots |
| `SynClassUtils.py` | Helpers shared by the scripts: reading and writing CSV files, the transformed HPO in the adapter's format, which every synonym file keeps (`readTransformed()`, `synonymTable()`, the gold synonym class of rows with `classesOf()`, the (concept, synonym) pairs with `keysOf()`), parsing the answers (`parseClassification()`), the check whether an output is up to date (`outdated()`, `writeSettings()`), the development and evaluation set (`writeDevelopmentSet()`), progress bars and the data summary |
| `SynClassConfig.py` | Pydantic schema of the `synclass` section of `config/config.yaml` and of its experiments |
| `config.py` | Reads the configuration, picks the experiment and the models from the command line, defines the column and class names, the file names of every step and the few-shot examples. `python3 ./src/config.py` prints the number of experiments |

The scripts import `config.py` and `SynClassUtils.py` with `from ... import *`. The packages of a step (`adapter`, `embedder`, `model`) are imported by the step itself, so that the preparation run does not load vLLM and the classification does not load the embedding models.

Each script starts with the logger: the first line of its log file is its command line, and every step is logged with its start, end and duration (`Logger.step()`).
