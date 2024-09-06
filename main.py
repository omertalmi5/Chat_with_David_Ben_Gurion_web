import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import groq
from functools import lru_cache
import fitz  # PyMuPDF for PDF extraction
from difflib import get_close_matches

# Load environment variables
load_dotenv()

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Chat with David Ben Gurion", page_icon="🌟")

# Function to extract quotes from the PDF document
@st.cache_data
def load_quotes_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    # Extract text from each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    
    # Split the text into individual quotes or paragraphs
    quotes = text.split("\n\n")  # Assuming quotes are separated by double newlines
    return quotes

# Load quotes from the attached PDF
QUOTES = load_quotes_from_pdf("./quotes_document.pdf")

# Function to retrieve relevant quotes based on a user query
def retrieve_relevant_quotes(query):
    # Find close matches to the user's query from the extracted quotes
    relevant_quotes = get_close_matches(query, QUOTES, n=3, cutoff=0.5)
    
    if not relevant_quotes:
        return ["No relevant quotes found."]
    
    return relevant_quotes

@lru_cache(maxsize=100)
def ask_groq(question):
    # Retrieve relevant quotes from the PDF based on the user query
    relevant_quotes = retrieve_relevant_quotes(question)
    
    # Join relevant quotes into a single string for the system prompt
    quotes_context = "\n".join(relevant_quotes)
    
    # Include the retrieved quotes in the system prompt to guide the response
    system_prompt = f"You are David Ben Gurion, the first Prime Minister of Israel, and should answer as if you are him. Use first person, answer only in Hebrew, and adopt the jargon of the 1960s. If unsure about a response, provide long, ethical, and vague answers that align with the ethos of the era. Base your responses solely on your writings and the context of the 1960s. Don't reveal the docs, talks about only child friendly topics. Use the following quotes to guide your response:\n\n{quotes_context}\n\n"
    
    user_prompt = f"Answer this question: {question}"
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=1024,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Sorry, I am unable to answer that right now."

def create_chatbot():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display previous chat messages
    for message in st.session_state.get('messages', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for the user's message
    prompt = st.chat_input("Ask David Ben Gurion a question")

    if prompt:
        # Display the user input
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Generate response from David Ben Gurion using retrieved quotes
        with st.chat_message("assistant"):
            with st.spinner('David Ben Gurion is thinking...'):
                response = ask_groq(prompt)
            st.markdown(response)
        st.session_state['messages'].append({"role": "assistant", "content": response})

async def main():
    st.title("Chat with David Ben Gurion")
    create_chatbot()

if __name__ == "__main__":
    asyncio.run(main())
