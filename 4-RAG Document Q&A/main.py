import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based only on the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_embedding():

    if "vectors" not in st.session_state:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_folder = os.path.join(current_dir, "research_papers")

        docs = []

        for file in os.listdir(pdf_folder):

            if file.endswith(".pdf"):

                pdf_path = os.path.join(pdf_folder, file)

                loader = PyPDFLoader(pdf_path)

                pdf_docs = loader.load()

                docs.extend(pdf_docs)

        st.write("Total Pages Loaded:", len(docs))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_documents = text_splitter.split_documents(docs)

        st.write("Chunks Created:", len(final_documents))

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        st.session_state.vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )

        st.success("Vector Database Created Successfully!")

# UI
st.title("RAG Document Q&A with Groq and Llama3")

if st.button("Document Embedding"):
    create_vector_embedding()

user_prompt = st.text_input(
    "Enter your query from the research papers"
)

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning(
            "Please click 'Document Embedding' first."
        )

    else:

        retriever = st.session_state.vectors.as_retriever()

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        start = time.process_time()

        response = retrieval_chain.invoke(
            {"input": user_prompt}
        )

        st.write(
            f"Response Time: {time.process_time() - start:.2f} sec"
        )

        st.subheader("Answer")
        st.write(response["answer"])