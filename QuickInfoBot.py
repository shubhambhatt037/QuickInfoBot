import re
import wikipedia
import spacy
import streamlit as st
from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def process_query(query):
    # Tokenize the query and extract entities using spaCy
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    main_subject = entities[0] if entities else query
    return main_subject

def get_wikipedia_summary(query):
    # Fetch the Wikipedia page summary for the given query
    try:
        summary = wikipedia.summary(query)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # If there are multiple results, select the first one
        page_title = e.options[0]
        summary = wikipedia.summary(page_title)
        return summary
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find information on that topic."

def main():
    st.title("Wikipedia Summarizer")

    # User input for the topic
    user_input = st.text_input("Enter Topic Here:", "")

    if st.button("Generate Summary"):
        if user_input:
            # Process user input
            main_subject = process_query(user_input)
            # Retrieve Wikipedia summary
            response = get_wikipedia_summary(main_subject)
            # Summarize the Wikipedia content
            summary_result = summarizer(response, max_length=1000, min_length=30, do_sample=False)

            # Display the Wikipedia summary and the summarized content
            st.subheader("Original Wikipedia Summary:")
            st.write(response)

            st.subheader("Summarized Content:")
            st.write(summary_result[0]['summary_text'])
        else:
            st.warning("Please enter a topic.")

if __name__ == '__main__':
    main()

