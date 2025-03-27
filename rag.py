import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig, TextStreamer

from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores import FAISS

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
    db = database.from_documents(chunked_docs, embeddings)
    db.save_local(path)

def load_index(database, embeddings, path, search_type='similarity', search_kwargs={"k": 4}):
    """
    Load index local

    Args:
    database: Vectordatabase
    embeddings: Embeddings to use
    path: Path to load index
    search_type (str):
        - 'mmr': maximum marginal relevance
        - 'similarity': take top k similar docs
        - 'similarity_score_threshold': sets a similarity score threshold and only returns documents with a score above that
    search_kwargs (dict):
        - {'k': int} Amount of documents to return (Default: 4)
        - {score_threshold: float (0..1)} Minimum relevance threshold (for search_type=similarity_score_threhsold)
        - {fetch_k: int}: Amount of documents to pass to MMR algorithm (default: 20)
        - {'lambda_mult': float (1 for min ... 0 for max)} Diversity of results returned by MMR (default: 0.5)
        - filter: Filter by metadata (e.g {'paper_title' : 'name of paper'}
    
    Return:
    retriever: Vector DB as retriever
    """
    db = database.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    return retriever

def load_prompt():
    """
    Load prompt from LangSmith Hub

    Args:
    -

    Return:
    rephrase_prompt: Prompt to rephrase the query to align with previous conversation
    retrieval_qa_chat_prompt: Prompt to insert context from retrieval and answer based on the context and the query
    """
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase", api_key=os.environ['LANGSMITH_API_KEY'], api_url="https://api.smith.langchain.com/")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat", api_key=os.environ['LANGSMITH_API_KEY'], api_url="https://api.smith.langchain.com/")
    # print("Rephrase prompt: ", rephrase_prompt)
    # print("\n")
    # print("Retrieval QA prompt: ", retrieval_qa_chat_prompt)

    return rephrase_prompt, retrieval_qa_prompt


def create_llm(model_name, as_streamer=True):
    """
    Load llm from HuggingFace and create ChatModel

    Args:
    model_name (str): HuggingFace model ID
    as_streamer (bool): if True, stream text during generation otherwise not

    Return:
    llm: ChatHuggingFace model
    """
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config, device_map="auto", token=os.environ['hf_token'])
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['hf_token'])
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if as_streamer == True:
        pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, streamer=streamer)
    else:
        pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    
    llm = HuggingFacePipeline(model_id=model_name, pipeline=pipe)
    llm = ChatHuggingFace(llm=llm)


    return llm

    
def create_rag(llm, rephrase_prompt, retrieval_qa_prompt, retriever):
    """
    Load RAG prompts from LangSmith Hub and create RAG chain (with history aware retriever)

    RAG Chain is based on (stuff_document_chain, history_aware_retriever_chain)
    - create_history_aware_retriever: Create a chain that takes conversation history and returns documents
    - create_stuff_documents_chain: Create a chain for passing a list of Documents to a model
    - create_retrieval_chain: Create retrieval chain that retrieves documents and then passes them on

    Args:
    llm: LLM to use for the answer
    rephrase_prompt: Prompt to rephrase the query to find the retrieval in vectordatabase (to get context to previous conversations)
    retrieval_qa_prompt: Prompt to get the retrieve the information from the vector database and formulate an answer
    retriever: Vectordatabase as a retriever

    Return:
    retrieval_chain: RAG chain from create_retrieval_chain()
    
    """
    # Create Chain
    # Create Document Stuff Chain
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # Create History aware Chain
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=rephrase_prompt)
    # Create Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)

    return retrieval_chain











