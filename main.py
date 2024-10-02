import streamlit as st
import tempfile
from embedchain import BotAgent
from embedchain.models import LLaMA
from embedchain.vector_stores import Weaviate
# Choose your LLM
llm = LLaMA()
# Choose your vector database
vector_store = Weaviate()
st.title("Chat with Your PDFs")
st.caption("A locally hosted LLM app with RAG for conversing with your PDF documents.")
# Create a temporary directory for the vector database
temp_dir = tempfile.mkdtemp()
# Create an instance of the Embedchain bot
bot = BotAgent(llm=llm, vector_store=vector_store, temp_dir=temp_dir)
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Create a temporary file and write the contents of the uploaded file to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
# Add the PDF file to the knowledge base
    bot.add_source(temp_file_path)
st.success(f"Successfully added {uploaded_file.name} to the knowledge base!")
question = st.text_input("Ask a question about the PDF:")
if question:
    try:
        answer = bot.query(question)
        st.write(answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")