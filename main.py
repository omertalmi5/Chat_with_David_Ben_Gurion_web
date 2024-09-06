import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from groq_client import Client
from functools import lru_cache

# Load environment variables
load_dotenv()

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Client(api_key=GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Chat with David Ben Gurion", page_icon="🌟")

@lru_cache(maxsize=100)
def ask_groq(question):
    # Set up the system prompt for David Ben Gurion
    system_prompt = "You are David Ben Gurion, the first prime minister of Israel. Answer questions in a way that reflects your personality, historical knowledge, and leadership style."
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

        # Generate response from David Ben Gurion
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
