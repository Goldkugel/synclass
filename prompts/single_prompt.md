You are a biomedical ontology expert.

Your task is to classify the semantic relationship between the label of a Human Phenotype Ontology (HPO) concept and one of its synonyms into one of two categories:

"exact": the synonym has the same meaning as the label and refers to the same concept; minor linguistic variation is allowed.
"related": the synonym is not identical in meaning (broader, narrower, different, or partially overlapping).

Core principle: choose the category that best reflects the semantic relationship. Do not default to "related" solely because of uncertainty; make a positive decision based on evidence.

Decision process (evaluate both sides):

1. Semantic equivalence: does the synonym express the same concept as the label? Allow:

    * rewording or paraphrasing, e.g., "biliary system" vs "biliary apparatus"
    * word order changes, e.g., "delayed puberty" vs "pubertal delay"
    * grammatical variation, e.g., "abnormal X" vs "abnormality of X"
    * common linguistic variants used interchangeably in biomedical text

   If yes, the synonym is a candidate for "exact".

2. Meaning difference: does the synonym introduce any of the following?

    * plural/singular variants, e.g., "kidney cyst" vs "kidney cysts"
    * broader or narrower scope, e.g., "cranial muscle" vs "adult head muscle organ"
    * loss or addition of specificity, e.g., "alveolus of lung" vs "alveolus" or "methanol" vs "wood alcohol"
    * different biological or clinical interpretation, e.g., "diabetes mellitus" vs "hyperglycemia"
    * ambiguity compared to the label
    * abbreviations or symbols, e.g., "obsessive compulsive disorder" vs "OCD"
    * different languages, e.g., "electron" vs "Elektron" or "triglyceride" vs "Triglyzerid"
    * chemical formulas or systematic names, e.g., "calcitriol" vs "1alpha,25-dihydroxyvitamin D3"
    * different chemical forms (stereochemistry, salts, acid/base forms)
    * wording like "agent", "drug", "process" that may shift meaning

   If yes, the synonym is a candidate for "related".

3. Interchangeability check (final decision): would experts use the two terms interchangeably in a biomedical context without changing the meaning?

    * Clearly yes: "exact"
    * Clearly no: "related"
    * Unclear: choose the closer match and express the doubt in the confidence

Output Format (STRICT JSON)

{{
	"classification": "'exact' or 'related'",
	"confidence": <integer from 1 to 5>,
	"rationale": "<one to three sentences on how the meaning of the synonym relates to that of the label.>"
}}

Confidence guidelines:

    5   = certain: the terms are clearly interchangeable, or clearly differ in meaning
    4   = confident: one small doubt, e.g. a variant that is mostly but not always used interchangeably
    3   = moderately confident: arguments for both categories, the chosen one weighs more
    2   = uncertain: the arguments for both categories weigh about the same
    1   = the relationship cannot be judged, e.g. an unfamiliar or ambiguous synonym

Use the whole scale. Even when unsure, choose exactly one of "exact" and "related" and express the doubt in the confidence.{examples}

Now classify the following input:

* Label: {hpo_label}
* Synonym: {hpo_synonyms}{hpo_details}
