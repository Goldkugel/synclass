Your last task:

Compare the biomedical meaning of the label and the synonym using your analyses from the previous steps, and classify the synonym into one of two categories: "exact" or "related".

Your goal is to determine whether the two terms represent:

    * the same biomedical concept ("exact")
    * or non-identical concepts ("related")

Classification rules:

* "exact":

    * rewording or paraphrasing, e.g., "biliary system" vs "biliary apparatus"
    * word order changes, e.g., "delayed puberty" vs "pubertal delay"
    * grammatical variation, e.g., "abnormal X" vs "abnormality of X"
    * common linguistic variants used interchangeably in biomedical text

* "related":

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

Decision rule:

* Compare the normalized concepts and semantic features from both analyses.
* Determine whether biomedical experts would use the terms interchangeably without changing the meaning.
* Make a positive semantic decision; do not default to "related" solely because of uncertainty.
* If uncertain, choose the closer match and reduce the confidence.

Output Format (STRICT JSON)

{{
	"rationale": "<one to three sentences on how the meaning of the synonym relates to that of the label.>",
	"classification": "'exact' or 'related'",
	"confidence": <integer from 1 to 5>
}}

Confidence guidelines:

    5   = certain: the terms are clearly interchangeable, or clearly differ in meaning
    4   = confident: one small doubt, e.g. a variant that is mostly but not always used interchangeably
    3   = moderately confident: arguments for both categories, the chosen one weighs more
    2   = uncertain: the arguments for both categories weigh about the same
    1   = the relationship cannot be judged, e.g. an unfamiliar or ambiguous synonym

Use the whole scale. Even when unsure, choose exactly one of "exact" and "related" and express the doubt in the confidence.{examples}
