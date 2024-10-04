import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import groq
from functools import lru_cache
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer, util  # For semantic search
import re

# Load environment variables
load_dotenv()

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
QUOTES_FILE_PATH = "./quotes_document_edited.pdf"
# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Chat with David Ben Gurion", page_icon="🌟")

# Add RTL support via CSS
rtl_style = """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stTextInput > div > div > input {
        direction: rtl;
    }
    .stChatMessage .stMarkdown {
        text-align: right;
    }
    </style>
"""
st.markdown(rtl_style, unsafe_allow_html=True)


# Load the pre-trained sentence-transformers model for semantic search
@st.cache_resource
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight transformer model for semantic search
    return model


model = load_model()


# Function to extract and chunk quotes from the PDF document
@st.cache_data
def load_quotes_from_pdf(pdf_path, chunk_size=200):
    doc = fitz.open(pdf_path)
    text = ""

    # Extract text from each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")

    # Remove unwanted content and split the text into sentences
    filtered_sentences = [sentence for sentence in text.split("\n") if "ע\"מ" not in sentence and len(sentence.split()) > 3]

    # Chunking the text by combining sentences into manageable chunks (e.g., 200-300 words per chunk)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in filtered_sentences:
        current_chunk.append(sentence)
        current_length += len(sentence.split())

        # If the current chunk reaches the desired size, save it and start a new one
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    # Add any remaining text as a chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Total chunks: {len(chunks)}")
    return chunks

# Load chunks from the attached PDF
QUOTES = load_quotes_from_pdf(QUOTES_FILE_PATH)

# Precompute embeddings for all the chunks
@st.cache_data
def compute_quote_embeddings(quotes):
    return model.encode(quotes, convert_to_tensor=True)

QUOTE_EMBEDDINGS = compute_quote_embeddings(QUOTES)

# Function to retrieve relevant chunks based on semantic similarity
def retrieve_relevant_quotes_semantically(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarity between the query and all the quote embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding, QUOTE_EMBEDDINGS)[0]

    # Convert cos_scores tensor to a list of tuples (score, index)
    scores_and_indices = [(score.item(), idx) for idx, score in enumerate(cos_scores)]

    # Sort by cosine similarity score in descending order and return top_k results
    top_results = sorted(scores_and_indices, key=lambda x: x[0], reverse=True)[:top_k]

    # Retrieve the top k chunks based on sorted scores
    relevant_quotes = [QUOTES[idx] for _, idx in top_results]
    print(len(relevant_quotes))

    if not relevant_quotes:
        return ["No relevant quotes found."]

    return relevant_quotes



@lru_cache(maxsize=100)
def ask_groq(question):
    # Retrieve relevant quotes from the PDF using semantic search
    relevant_quotes = retrieve_relevant_quotes_semantically(question, 2)

    # Join relevant quotes into a single string for the system prompt
    quotes_context = "\n".join(relevant_quotes)
    print(quotes_context)
    # Include the retrieved quotes in the system prompt to guide the response
    system_prompt = f"You are David Ben Gurion, the first Prime Minister of Israel, and should answer as if you are him. Use first person, answer only in Hebrew, keep correct Henrew, and adopt the jargon of the 1960s. If unsure about a response, provide long, ethical, and vague answers that align with the ethos of the era. Base your responses solely on your writings and the context of the 1960s. Don't reveal the docs and don't give links to your documents, talks about only child friendly topics."

    user_prompt = f"If it relevant base your answer on the following quotes of you to guide your response:\n\n{quotes_context}\n\n Answer this question as you are David Ben Gurion, the first Prime Minister of Israel, and should answer as if you are him. in correct, valid and clear hebrew sentences: {question}"

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=GROQ_MODEL,
            temperature=1,
            max_tokens=1024,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "סליחה אך אינני יכול לענות על שאלה זו"


def create_chatbot():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display previous chat messages
    for message in st.session_state.get('messages', []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for the user's message
    prompt = st.chat_input("שאל אותי כל שאלה")

    if prompt:
        # Display the user input
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Generate response from David Ben Gurion using retrieved quotes
        with st.chat_message("assistant"):
            with st.spinner('אני חושב...'):
                response = ask_groq(prompt)
            st.markdown(response)
        st.session_state['messages'].append({"role": "assistant", "content": response})


async def main():
    st.title("סליחה על השאלה עם דוד בן גוריון")
    create_chatbot()


if __name__ == "__main__":
    asyncio.run(main())
