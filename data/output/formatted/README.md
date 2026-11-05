# Formatted classifications

Written by the classification (`python3 ./src/SynClassClassification.py <index>`) after a model has classified all synonyms: one file per experiment and model, `class_<experiment>_formatted_<model>.csv`, read by `SynClassMerge.py`.

The columns are those of the raw classification (see [`../class/README.md`](../class/README.md)), except that `answer` holds the classification (`exact`, `related` or `undefined`) and `confidence` its confidence (`-1` if undefined); the `classification` column is left out. Wrong and undefined classifications are listed in the log with their confidence and rationale, followed by a summary of the model (gold against classified classes, precision, recall and F1), see [`../../logs/README.md`](../../logs/README.md).
