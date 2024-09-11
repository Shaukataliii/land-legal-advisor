from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument
from google.auth.exceptions import DefaultCredentialsError


@st.cache_data
def is_api_key_valid(api_key):
    print("Running src/main is_api_key_valid()")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content("How are you")
        return True
    except InvalidArgument or DefaultCredentialsError:
        return False
    except Exception as e:
        return False


class LLMHandler:
    def __init__(self):
        self.api_key_varname = "GOOGLE_API_KEY"
        self.llm_temperature = 0.3
        self.google_chatmodel_name = "gemini-1.5-flash"

    def load_api_key(self):
        load_dotenv()
        return os.getenv(self.api_key_varname)
        
    def load_google_llm(self, api_key):
        self.configure_genai(api_key)
        model = genai.GenerativeModel(self.google_chatmodel_name)
        return model
    
    def configure_genai(self, api_key):
        genai.configure(api_key=api_key)
    
    def get_embeddings_obj(self):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")


class DocumentPreparer:
    def __init__(self):
        self.separators_for_docs_splitting = ['\n\n', '\n', '.', ',']
        self.vdb_filepath = "resouces/vdb"

    def load_pdf_as_docs(self, pdf_filepath):
        if self.is_path_valid(pdf_filepath):
            loader = PDFMinerLoader(pdf_filepath)
            docs = loader.load()
            return docs
        
        else:
            raise Exception(f"Provided pdf filepath doesn't exist. {pdf_filepath}")
            
    def is_path_valid(self, path):
        if os.path.exists(path):
            return True
        else:
            return False
        
    def split_pdf_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter( separators=self.separators_for_docs_splitting )
        docs = splitter.split_documents(docs)
        return docs

    def create_save_chroma_vdb(self, documents, embeddings_obj):
        vector_db = Chroma.from_documents(documents, embeddings_obj, persist_directory=self.vdb_filepath)
        return vector_db
    
    def create_retriever_from_vdb(self, vdb):
        return vdb.as_retriever()
    
    def load_chroma_vdb(self, chroma_vdb_path):
        pass
        

class PromptFormatter:
    def format_context(self, context):
        return "\n\n".join([doc.page_content for doc in context])

    def get_str_output_parser(self):
        return StrOutputParser()