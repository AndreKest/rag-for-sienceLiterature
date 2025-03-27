import numpy as np
import pandas as pd

from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.schema import HumanMessage, AIMessage

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig, TextStreamer

import torch

import os
from dotenv import load_dotenv

load_dotenv()

def create_embeddings(embedding_name: str = "BAAI/bge-base-en-v1.5"):
    """
    Load HuggingFace embeddings

    Args:
    embedding_name (str): Name of the embeddings

    Return:
    embeddings: Loaded embeddings from HuggingFace
    """
    # Embeddings
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        #model_kwargs=model_kwargs,
        #encode_kwargs=encode_kwargs,
        cache_folder = './hf'
    )

    return embeddings

def create_chunks(docs, chunk_size=2048, chunk_overlap=30, seperators=None):
    """
    Create Chunks of docs

    Args:
    chunk_size (int): Size of each chunk
    chunk_overlap (int): Size of the overlap to previous chunk
    seperators (list): Try to chunk at characters from seperators

    Return:
    chunked_docs: Chunked documents
    """
    if seperators is None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, seperators=seperators)
        
    
    chunked_docs = splitter.split_documents(docs)

    return chunked_docs


def create_index(database, embeddings, chunked_docs, path):
    """ 
    Create index and save local

    Args:
    database: Vectordatabase
    embeddings: Embeddings to use
    chunked_docs: Chunked documents
    path: path to save the index

    Return:
    None
    """
    db = FAISS.from_documents(chunked_docs, embeddings)
    db.save_local(path)

def load_index(database, embeddings, path):
    """
    Load index local

    Args:
    database: Vectordatabase
    embeddings: Embeddings to use
    path: Path to load index
    
    Return:
    retriever: Vectordatabase as retriever
    """
    db = database.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return retriever

def load_


def create_rag(llm, rephrase_prompt, retrieval_qa_prompt, retriever):
    """
    Load RAG prompts from LangSmith Hub and create RAG chain (with history aware retriever)

    RAG Chain is based on (stuff_document_chain, history_aware_retriever_chain)

    TODO: Explain functions
    create_history_aware_retriever:
    create_stuff_documents_chain:
    create_retrieval_chain:

    Args:
    llm: LLM to use for the answer
    rephrase_prompt: Prompt to rephrase the query to find the retrieval in vectordatabase (to get context to previous conversations)
    retrieval_qa_prompt: Prompt to get the retrieve the information from the vector database and formulate an answer
    retriever: Vectordatabase as a retriever

    Return:
    retrieval_chain: Created RAG chain from create_retrieval_chain()
    
    """
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase", api_key=os.environ['LANGSMITH_API_KEY'], api_url="https://api.smith.langchain.com/")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat", api_key=os.environ['LANGSMITH_API_KEY'], api_url="https://api.smith.langchain.com/")
    print("Rephrase prompt: ", rephrase_prompt)
    print("\n")
    print("Retrieval QA prompt: ", retrieval_qa_chat_prompt)

    # Create Chain
    # Create Document Stuff Chain
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # Create History aware Chain
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=rephrase_prompt)
    # Create Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)

    return retrieval_chain











