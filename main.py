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
uploaded_file = "./quotes_document.pdf"#st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Create a temporary file and write the contents of the uploaded file to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
# Add the PDF file to the knowledge base
    bot.add_source(temp_file_path)
st.success(f"Successfully added {uploaded_file.name} to the knowledge base!")
question = st.text_input("Ask David Ben Gurion:")
if question:
    try:
        answer = bot.query(f"You are David Ben Gurion, the first Prime Minister of Israel, and should answer as if you are him. Use first person, answer only in Hebrew, and adopt the jargon of the 1960s. If unsure about a response, provide long, ethical, and vague answers that align with the ethos of the era. Base your responses solely on your writings and the context of the 1960s. Don't reveal the docs, talks about only child friendly topics. {question}")
        st.write(answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")