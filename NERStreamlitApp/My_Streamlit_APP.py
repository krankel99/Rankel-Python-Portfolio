# import the necessary libraries and packages
import pandas as pd
import streamlit as st
from spacy.pipeline import EntityRuler
import spacy


st.title('Custom NER Application') # adds a title to the app

# this loads the spaCy pipeline
nlp=spacy.load('en_core_web_sm')

# The EntityRuler is a spaCy component that allows us to add custom rules for entity recognition
# this removes an existing entity_ruler if present to avoid duplicating patterns
if 'entity_ruler' in nlp.pipe_names:
    nlp.remove_pipe('entity_ruler')
ruler=nlp.add_pipe('entity_ruler', last=True)


# this is using session_state to help keep track of custom patterns that the user enters
if 'custom_patterns' not in st.session_state:
    st.session_state.custom_patterns=[] # initializes a new list for custom patterns


st.header('Define Custom Entity Patterns') # display a header to indicate where custom patterns are defined

# this provides separate inputs for the entity label and the matching pattern text
custom_label=st.text_input('Entity Label', placeholder='e.g., ORG')
custom_pattern=st.text_input('Entity Pattern (text)', placeholder='e.g., Google')

# this uses a button to add a new custom pattern to session state
if st.button('Add Pattern'):
    if custom_label.strip()=="" or custom_pattern.strip()=="":
        st.error('Both label and pattern are required!') # this message appears if both inputs are not provided
    else:
        st.session_state.custom_patterns.append({
            'label': custom_label.strip(),
            'pattern': custom_pattern.strip
        })
        st.success(f'Added pattern: [{custom_label.strip()} -> {custom_pattern.strip()}]') # this is the message if it is a success

st.subheader('Current Custom Patterns:')

# this loops through the stored custom patterns and displays each one
if st.session_state.custom_patterns:
    for i, pat in enumerate(st.session_state.custom_patterns, start=1):
        st.write(f'{i}. Label: {pat['label']}, Pattern: {pat['pattern']}')
else:
    st.write('No custom patterns added yet.')

# this is a button to apply the custom patterns to the spaCy EntityRuler component
if st.button('Clear All Patterns'):
    ruler.patterns=[]
    for pat in st.session_state.custom_patterns:
        ruler.add_patterns([pat])
    st.success('Custom patterns have been applied!')

st.header('Input Text for NER Analysis')
# this is just a default text I provided, an interesting fact
default_text=(
    'The fastest mile time is 3 minutes and 43 seconds by Hicham El Guerrouj in Rome.'
)
user_text=st.text_area('Enter or paste text to analyze:', value=default_text, height=150)

doc=nlp(user_text)

# loop through each entity detected by spaCy
st.header('Detected Named Entities')
if doc.ents:
    for ent in doc.ents:
        st.write(f'{ent.text}-{ent.label_}')
else:
    st.write('No entities detected in the provided text.')

