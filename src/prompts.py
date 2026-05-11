import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

# Import necessary modules and configuration settings
from config import *
from utils import *

def createJSONExample(cl : str = "", co : str = "10") -> str:
    ret = "{\n" \
        f"\t{quote(synonymClass)}: {quote(cl)},\n" \
        f"\t{quote(confidenceColumn)}: {co}\n" \
        "}"
    
    return ret

def formatInput(label : str = "", definition : str = "", comment : str = "", parents : list = [], children : list = [], synonym : str = "") -> str:
    ret = f"Label: {quote(label)}"

    if addDefinition:
        ret = ret + f"\nDefinition: {quote(definition)}"
    if addComment:
        ret = ret + f"\nComment: {quote(comment)}"
    if addParents:
        ret = ret + f"\nParents: {applyFormat(parents)}"
    if addChildren:
        ret = ret + f"\nChildren: {applyFormat(children)}"
    if len(synonym) > 0:
        ret = ret + f"\nSynonym: {quote(synonym)}"

    return ret

def getExamples() -> str:
    return f"""Examples

Example 1

Input: 
{formatInput("Neck muscle hypoplasia", "Underdevelopment of muscles of the neck.", "", ["Abnormal neck morphology", "Hypoplasia of the musculature"], [], "Underdevelopment of neck muscle")}

Output:
{createJSONExample("exact", "9")}

Example 2

Input:
{formatInput("D-cysteine", "An optically active form of cysteine having D-configuration.", "", ["cysteine", "D-α-amino acid"], [], "DCY")} 

Output:
{createJSONExample("related", "8")}

Example 3

Input:
{formatInput("mandibular ramus", "The upturned perpendicular extremity of the mandible.", "", ["zone of bone organ"], ["mandible condylar process", "mandible coronoid process", "mandible temporal crest"], "rami mandibulae")} 

Output:
{createJSONExample("related", "10")}"""

def semanticClassificationPrompt1(
        label : str, 
        definition : str, 
        comment : str,  
        parents : list,
        children : list
) -> str:
    return f"""You are a biomedical ontology expert.

Your fist task:

Analyze the Label term and identify its biomedical meaning.

Instructions:

1. Normalize the Label into standard biomedical language.
2. Identify the precise biomedical concept represented by the Label.
3. Extract important semantic properties.

Focus especially on:

    * biological or clinical meaning
    * chemical structure interpretation
    * acid/base state
    * ionic state
    * stereochemistry
    * salt forms
    * specificity level
    * anatomical scope
    * process vs entity distinctions

Ignore superficial linguistic variation such as:

    * word order
    * punctuation
    * hyphenation
    * grammatical variation

Input:

{formatInput(label, definition, comment, parents, children, "")}"""

def semanticClassificationPrompt2(
    synonym : str
) -> str:
    return f"""Your second task:

Analyze ONLY the Synonym term and identify its biomedical meaning.

Instructions:

1. Normalize the Synonym into standard biomedical language.
2. Identify the precise biomedical concept represented by the Synonym.
3. Extract important semantic properties.

Focus especially on:

    * biological or clinical meaning
    * chemical structure interpretation
    * acid/base state
    * ionic state
    * stereochemistry
    * salt forms
    * specificity level
    * anatomical scope
    * process vs entity distinctions

Ignore superficial linguistic variation such as:

    * word order
    * punctuation
    * hyphenation
    * grammatical variation

Input:

Synonym: {quote(synonym)}"""

def semanticClassificationPrompt3(fewShot : bool = fewShot) -> str:
    ret = f"""Your third Task:

Compare the biomedical meaning of the Label and the Synonym using your prior semantic analyses.

Your goal is to determine whether the two terms represent:

    * the SAME biomedical concept ("exact")
    * or NON-identical concepts ("related")

Classification Rules:

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

Instructions:

1. Compare the normalized concepts and semantic features from both analyses.
2. Determine whether biomedical experts would use the terms interchangeably without changing meaning.
3. Make a positive semantic decision.
4. Do NOT default to "related" solely because of uncertainty.
5. If uncertain, choose the closer match and reduce confidence.

Output Format (STRICT JSON)

{createJSONExample('"exact" or "related"', '<integer from 1 to 10>')}

Confidence guidelines:

    * 10 → completely certain
    * 7–9 → high confidence
    * 4–6 → moderate uncertainty
    * 1–3 → low confidence / guess

Do not include any additional text."""

    if fewShot:
        ret = ret + "\n\n" + getExamples()

    return ret

def semanticClassificationPrompt(
        label : str, 
        definition : str, 
        comment : str, 
        parents : list, 
        children : list,
        synonym : str,
        fewShot : bool = fewShot
)-> str:
    ret = \
f"""You are a biomedical ontology expert.

Your task:

Classify the semantic relationship between a Label and a Synonym.

Classes:

    * "exact": same meaning and refers to the same concept; minor linguistic variation is allowed.
    * "related": not identical in meaning (broader, narrower, different, or partially overlapping).

Core Principle:
Choose the class that BEST reflects the semantic relationship.
Do NOT default to "related" solely due to uncertainty. Make a positive decision based on evidence.

Decision Process (evaluate both sides):

1. Semantic Equivalence
   Does the synonym express the SAME concept as the label?
   Allow:

    * rewording or paraphrasing, e.g., "biliary system" vs "biliary apparatus"
    * word order changes, e.g., "delayed puberty" vs "pubertal delay"
    * grammatical variation, e.g., "abnormal X" vs "abnormality of X"
    * common linguistic variants used interchangeably in biomedical text

If YES → candidate for "exact"

2. Meaning Difference
   Does the synonym introduce ANY of the following?

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

If YES → candidate for "related"

3. Interchangeability Check (final decision)
   Would experts use these terms interchangeably in biomedical context without changing meaning?

    * Clearly yes → "exact"
    * Clearly no → "related"
    * Unclear → choose the closer match and reflect uncertainty in confidence

Output format (STRICT JSON):

{createJSONExample("'exact' or 'related'", "<integer from 1 to 10>")}

Confidence guidelines:

    * 9–10 = clearly correct
    * 6–8 = reasonably confident
    * 3–5 = uncertain
    * 1–2 = weak guess

"""
    
    if fewShot:
        ret = ret + getExamples() + "\n\n"

    ret = ret + f"""Now classify the following input:

{formatInput(label, definition, comment, parents, children, synonym)}"""

    return ret