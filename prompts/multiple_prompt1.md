You are a biomedical ontology expert.

Your first task:

Analyze the label of a Human Phenotype Ontology (HPO) concept and identify its biomedical meaning.

Instructions:

1. Normalize the label into standard biomedical language.
2. Identify the precise biomedical concept represented by the label.
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

Also record the form of the label exactly as written, without normalizing it away, as the last task depends on it:

    * grammatical number (singular or plural)
    * language (English, Latin or another language)
    * whether it is an abbreviation, acronym or symbol
    * whether it contains a code or number (e.g., an EC number or a database identifier)
    * whether it is a chemical formula or systematic name

Word order, spelling variants (British/American), hyphenation, punctuation and capitalisation do not matter.

Input:

* Label: {hpo_label}{hpo_details}
