import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG Document Chatbot")
st.markdown("Upload a PDF document and ask questions about its content. This session uses a fresh database!")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if st.button("Clear Session"):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.current_file = None
        st.rerun()

    if uploaded_file:
        # If a new file is uploaded, process it
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = [] # Clear chat history
            
            with st.spinner("Processing document... (This might take a moment)"):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load and split
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)
                    
                    # Create embeddings and vectorstore (in-memory, no persist_directory)
                    embedding_model = HuggingFaceEmbeddings()
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model
                    )
                    
                    # Create retriever
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 3, 
                            "fetch_k": 10, 
                            "lambda_mult": 0.5
                        }
                    )
                    st.success("Document processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                finally:
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)

# --- Main Chat Interface ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if query := st.chat_input("Ask a question about your document"):
    if not st.session_state.retriever:
        st.warning("Please upload a document first from the sidebar.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Setup LLM and Prompt
        llm = ChatMistralAI(model="mistral-small-latest")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful assistant that provides concise answers based on retrieved documents. If the retrieved documents do not contain enough information to answer the question, say "I could not find the information in the given document" """),
                ("human", "Context:{context}\n\nQuestion:{question}")
            ]
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve documents
                    docs = st.session_state.retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate response
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })
                    
                    response = llm.invoke(final_prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")
