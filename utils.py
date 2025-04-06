import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_groq import ChatGroq


def load_documents(dir: str):
    loader = PyPDFDirectoryLoader(
        dir,
        glob = '**/[!.]*.pdf',
        extract_images = False
    )
    documents = loader.load()

    return documents


def split_documents(docs, ChunkSize=1024, ChunkOverlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ChunkSize,
        chunk_overlap=ChunkOverlap
    )
    text_chunks = text_splitter.split_documents(docs)

    return text_chunks


def load_embeddings(
    embd_model_name=None,
    embd_type='GoogleGenAI',
    device='cpu'  # or 'cuda' if GPU is available and desired
):
    if embd_type == 'GoogleGenAI':
        if embd_model_name is None:
            embd_model_name = "models/text-embedding-004"
        embeddings = GoogleGenerativeAIEmbeddings(model=embd_model_name)

    elif embd_type == 'HuggingFace':
        if embd_model_name is None:
            embd_model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=embd_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False},
            cache_folder=os.getcwd(),
            show_progress=True
        )

    elif embd_type == 'OpenAI':
        if embd_model_name is None:
            embd_model_name = "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(model=embd_model_name)

    else:
        raise ValueError(f"Unsupported embd_type: {embd_type}")
    
    return embeddings


def initialize_llm(model_name=None, temp=0.6):
    if model_name is None:
        model_name = "deepseek-r1-distill-llama-70b"
    llm = ChatGroq(
        model = model_name,
        temperature = temp,
    )
    return llm