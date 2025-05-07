# NER Application
## Project Overview
This Streamlit app lets users explore Named Entity Recognition (NER) using the spaCy NLP library. It combines spaCy’s pretrained statistical NER model with a rule-based `EntityRuler`, allowing users to input or upload custom text, define their own entity patterns, and instantly view highlighted entities. The app requires no coding and enables real-time visualization using spaCy’s `displaCy` tool. This project demonstrates how rule-based and statistical NLP techniques can work together to provide flexible, explainable entity recognition.

## Instructions
- Do `pip install -r requirements.txt`

- Then `streamlit run My_Streamlit_App.py`

- View deployed app: https://rankel-python-portfolio-fjaypphs96jea4q6tqqclo.streamlit.app/
## App Features
**Flexible Text Input**: Upload, paste or type any text you'd like to analyze.

**Custom Entity Patterns**: Add your own entity rules using a label (e.g., ORG, PERSON) and a simple pattern string (e.g., "Google).

**Visual Highlighting**: Uses displaCy to render entities inline with distinct colors.

**Clear and Reusable**: Easily clear all patterns and try new ones all with the push of a button.
## References
SpaCy Library Architect: https://spacy.io/api

The Basics of spaCy: https://spacy.pythonhumanities.com/01_01_install_and_containers.html

SpaCy and its Linguistic Annotations: https://spacy.pythonhumanities.com/01_02_linguistic_annotations.html

Using SpaCy's EntityRuler: https://spacy.pythonhumanities.com/02_01_entityruler.html
## Visual Examples
![image](https://github.com/user-attachments/assets/a7cca8dd-d336-460d-b81d-2ab53f48e197)

![image](https://github.com/user-attachments/assets/2eef9bf1-ff59-4e3b-a91e-3260d43446a2)


