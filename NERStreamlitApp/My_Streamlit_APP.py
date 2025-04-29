# import the necessary libraries and packages
import pandas as pd
import streamlit as st
from spacy.pipeline import entityruler
import spacy
from spacy import displacy

# Initializes session state for custom patterns
# st.session_state is a built-in dictionary-like object in Streamlit that allows one to persist data across user interactions like button clicks and form submissions
# I use it mainly for the custom_patterns as I want the list to keep growing even as the script loops back after each button click
if 'custom_patterns' not in st.session_state:
 # creates a list to hold custom patterns across app interactions
    st.session_state.custom_patterns = []

if 'label_colors' not in st.session_state:
    # creates a dictionary to hold colors for each label
    st.session_state.label_colors = {}

# Set up the spaCy pipeline and EntityRuler
nlp = spacy.load('en_core_web_sm') # this is a pre-trained English model
if 'entity_ruler' in nlp.pipe_names:
    # removes existing ruler to avoid duplicating rules on rerun
    nlp.remove_pipe('entity_ruler')
ruler = nlp.add_pipe('entity_ruler', last=True) # adds a fresh EntityRuler at the end

# Reapply stored custom patterns to the EntityRuler
for pat in st.session_state.custom_patterns:
    ruler.add_patterns([pat])

# this is the start of what is actually displayed on the streamlit app
# this section is for inputing text for analysis
st.title('Custom NER Application') # this is the app title displayed at the top

st.header('Input Text for NER Analysis') # section header
# this provides an options for file uploader but only for TXT files
uploaded_file = st.file_uploader("Upload a text file (TXT only)", type=["txt"])

if uploaded_file is not None:
    # If user uploads a file, read the file and decode it to a UTF-8 string
    user_text = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully!")
else:
    # Otherwise show a text area for manual text input with a default example about the fastest mile
    default_text = (
        'The fastest mile time is 3 minutes and 43 seconds by Hicham El Guerrouj in Rome.'
    )
    user_text = st.text_area('Enter or paste text to analyze:', value=default_text, height=150)

# Process text with current pipeline
doc = nlp(user_text)

# this section displays the raw entity list
if doc.ents:
    # Loop through each entity detected by spaCy
    for ent in doc.ents:
        # Display the entity text and its corresponding label in a formatted manner
        st.write(f"**{ent.text}** — {ent.label_}")
else:
    # Inform the user if no entities were detected
    st.write("No entities detected in the provided text.")

# this section is for the highlighted visualization
st.header("Highlighted Entities")
# use HTML with entities highlighted, then display
# HTML stands for HyperText Markup Language and it is used to structure content on the web
html = displacy.render(doc, style="ent")
st.markdown(html, unsafe_allow_html=True)

# this section is for custom pattern definitions
st.header('Define Custom Entity Patterns')

# Inputs for custom entity label and pattern
custom_label = st.text_input('Entity Label', placeholder='e.g., ORG')
custom_pattern = st.text_input('Entity Pattern (text)', placeholder='e.g., Google')

# Button to add a new custom pattern
if st.button('Add Pattern'):
    # validates that the inputs are non-empty
    if custom_label.strip() == "" or custom_pattern.strip() == "":
        st.error('Both label and pattern are required!')
    else:
        # append new pattern dictionary to session state list
        st.session_state.custom_patterns.append({
            'label': custom_label.strip(),
            'pattern': custom_pattern.strip()
        })
        st.success(f'Added pattern: [{custom_label.strip()} -> {custom_pattern.strip()}]')

# Display the list of current custom patterns
st.subheader('Current Custom Patterns:')
if st.session_state.custom_patterns:
    for i, pat in enumerate(st.session_state.custom_patterns, start=1):
        st.write(f"{i}. Label: {pat['label']}, Pattern: {pat['pattern']}")
else:
    st.write('No custom patterns added yet.')


# this section clears all the patterns button
if st.button('Clear All Patterns'):
    # Clear the stored custom patterns and label colors
    st.session_state.custom_patterns = [] 
    st.session_state.label_colors = {}
    
    # Remove and re-add the EntityRuler to reset its patterns
    if 'entity_ruler' in nlp.pipe_names:
        nlp.remove_pipe('entity_ruler')
    ruler = nlp.add_pipe('entity_ruler', last=True)
    
    st.success('All custom patterns have been cleared!')


