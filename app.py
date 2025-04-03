import streamlit as st
import tempfile
import os
from agno.agent import Agent, RunResponse
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama

# Streamlit UI
st.title("📄 Ollama PDF Chatbot 🤖")
st.write("Upload a PDF and start asking questions!")

# File Uploader
file = st.file_uploader("Upload a PDF file", type=["pdf"])

if file is not None:
    st.write("✅ PDF uploaded successfully! Now, ask your question.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        pdf_path = temp_file.name
    
    # Load PDF into Knowledge Base with Persistent ChromaDB
    pdf_knowledge_base = PDFKnowledgeBase(
        path=pdf_path,
        vector_db=ChromaDb(
            collection="pdf_chatbot",
            embedder=OllamaEmbedder(id="openhermes"),
            persistent_client=True, # Specify persistent storage
        ),
        reader=PDFReader(chunk=True),
    )
    pdf_knowledge_base.load(recreate=False)  # Load into ChromaDB
    
    # Create the AI Agent
    agent = Agent(
        model=Ollama(id="llama3.2"),  # Ollama model
        knowledge=pdf_knowledge_base,
        search_knowledge=True,
        markdown=True
    )
    
    # Chat Input
    user_query = st.chat_input("Ask a question about the PDF...")
    
    if user_query:
        st.write(f"**You:** {user_query}")
        
        # Query the agent
        run: RunResponse = agent.run(user_query)
        st.write(f"**Bot:** {run.content}")

    # Cleanup temp file
    os.unlink(pdf_path)
else:
    st.write("⚠️ Please upload a PDF file to start the chat.")
