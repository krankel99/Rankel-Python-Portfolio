### Introduction
This interactive web application allows users to explore and experiment with Named Entity Recognition (NER) using the spaCy NLP library. It was built with the Streamlit app that lets users input or upload custom text, define their own entity patterns, and instantly view recognized entities without writing any code.
### Project Overview
NER is a core NLP task that involves identifying and classifying entities in text (like names, locations, organizations, etc).

This app leverages spaCy's EntityRuler, which allows users to define rule-based entity patterns using token-level or phrase-based matching and integrates them into spaCy's processing pipeline. The goal of this project is to make it easy for users to test custom NER logic, visualize results, and better understand how entity recognition works in spaCy.
### Instructions
Install the necessary libraries and packages like spaCy, streamlit and pandas.

Make sure everything is upgraded as well.

Download "python -m spacy download en_core_web_sm" for spaCy's English model.

Streamlit run app.py

Local host: [http://localhost:8501/](url)
### App Features
**Flexible Text Input**: Upload, paste or type any text you'd like to analyze.

**Custom Entity Patterns**: Add your own entity rules using a label (e.g., ORG, PERSON) and a simple pattern string (e.g., "Google).

**Visual Highlighting**: Uses displaCy to render entities inline with distinct colors.

**Clear and Reusable**: Easily clear all patterns and try new ones all with the push of a button.
### References
SpaCy Library Architect: [https://spacy.io/api](url)

The Basics of spaCy: [https://spacy.pythonhumanities.com/01_01_install_and_containers.html](url)

SpaCy and its Linguistic Annotations: [https://spacy.pythonhumanities.com/01_02_linguistic_annotations.html](url)

Using SpaCy's EntityRuler: [https://spacy.pythonhumanities.com/02_01_entityruler.html](url)
### Visual Examples
![image](https://github.com/user-attachments/assets/488f072e-dddc-45e5-afc0-a940a82f82eb)

![image](https://github.com/user-attachments/assets/057187ca-0747-4d71-ac00-4130edab964f)

