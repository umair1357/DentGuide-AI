import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(page_title="PDF Q&A with Gemini", page_icon="📚")
st.title("📚 Ultra-Lean PDF Q&A Bot")
st.caption("Upload PDFs and query them efficiently using Google Gemini.")

# -------------------------------------------------------------------
# API Key Verification
# -------------------------------------------------------------------
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if not api_key and "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

if not api_key:
    st.info("Please enter your Gemini API key in the sidebar or set `GOOGLE_API_KEY` in Streamlit Secrets.", icon="🔑")
    st.stop()


# -------------------------------------------------------------------
# Helper Functions (Cached using raw file bytes to prevent HashErrors)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Processing and indexing PDF...")
def process_pdf_bytes(file_bytes, file_name, key):
    """Loads PDF bytes, splits into chunks, and builds a FAISS vector index."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # 1. Load document
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 2. Split text into small chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Create Vector Store passing explicit API Key
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-001",
            google_api_key=key
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------------------------------------------------------------------
# File Upload & Chat
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Pass raw bytes to cached processor to prevent Streamlit HashError
    vectorstore = process_pdf_bytes(
        uploaded_file.getvalue(), 
        uploaded_file.name, 
        api_key
    )
    st.success("PDF processed successfully!")

    user_query = st.chat_input("Ask a question about the document...")

    if user_query:
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching relevant context..."):
                # Retrieve ONLY top 3 chunks to conserve tokens
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_query)

                context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

                template = """You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the answer cannot be deduced from the context, state "I couldn't find the answer in the provided document." Keep the response concise.

Context:
{context}

Question:
{question}

Answer:"""
                
                prompt = PromptTemplate.from_template(template).format(
                    context=context_text,
                    question=user_query
                )

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.2
                )
                
                response = llm.invoke(prompt)
                st.write(response.content)

                with st.expander("View retrieved context (Token saving debug)"):
                    st.text(context_text)
