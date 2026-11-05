# Evaluation

Written by `python3 ./src/SynClassEval.py <index>` from the merged classifications of an experiment (`class_<experiment>_...`). Only synonyms whose gold synonym class is `exact` or `related` are evaluated, and the synonyms of the few-shot examples are left out, also in runs without examples.

| File | Content |
|---|---|
| `..._gold_counts.png` | Number of synonyms per gold synonym class |
| `..._answer_counts.png` | Number of classifications per synonym class and model |
| `..._base_evaluation.png` | Precision, recall and F1 per model and synonym class |
| `..._summary.csv` | One row per model and baseline (`majority (<class>)`: always the most frequent gold class; `distance (<model>)`: the distance threshold of the development set): macro-F1, balanced accuracy, Cohen's kappa, accuracy and the share of undefined answers, each with the bounds of its 95% interval from `synclass.bootstrap_samples` bootstrap samples (1000 by default; `_low`, `_high`); also logged |
| `..._confusion_matrices.png` | Gold class × answer (exact, related) per model, as counts and as shares of all synonyms of the gold class; undefined answers are left out (their share is in `..._summary.csv`) |
| `..._roc.png` | ROC curves and AUC with related as the positive class: a model ranks the synonyms by its confidence (positive for related answers, negative for exact ones), the baseline by the distance; the AUCs are logged |
| `..._calibration.png` | Accuracy and share of the answers per confidence level (1 to 5) of every model; the accuracies are logged |
| `class_<Test>_distance_evaluation_distance_<model>_cosine.png` | Density of the normalised distances (see [`../embedded/README.md`](../embedded/README.md)) per gold synonym class |
| `class_<Test>_embedding_evaluation_distance_<model>_cosine.png` | The distance baseline on the development set: macro-F1 and the F1 per class of classifying by one threshold (related if the distance is at least the threshold, i.e. further from the label; exact otherwise), per threshold within `synclass.distance_evaluation_lower_bound` and `_upper_bound` (widened to the data), one every `synclass.distance_evaluation_step`, with the best threshold marked |

With a threshold in `synclass.distance_thresholds`, the distance classifier `<model> (distance)` (related at or above the threshold, exact below) is evaluated like the LLMs: in the summary, the confusion matrices, the answer counts and the precision, recall and F1 plot; having no confidence, it is not in the calibration plot, and the ROC curves rank by the distance itself. It then replaces the `distance (<model>)` baseline in the summary.

The distance baseline chooses its threshold on the development set by macro-F1, so that both classes count the same whatever their share. Runs with `test` are scored on the development set itself (the threshold is chosen on the same synonyms); the other runs apply the threshold to the evaluation set (`embed.development.csv` gives it). The threshold and the scores are logged for every run; the distance plots are drawn for experiments with `test`, or for all with `synclass.force_draw_additional_plots`, and drawn again on every run.

The logs also give the number of synonyms per ontology and, per model, the share of undefined answers: an undefined answer lowers the recall of its gold class but no precision.
