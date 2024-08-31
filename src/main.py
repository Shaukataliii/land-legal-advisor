from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LLMHandler:
    def __init__(self):
        self.api_key_varname = "GOOGLE_API_KEY"

    def is_api_key_valid(self, api_key):
        pass
    

    def load_api_key(self):
        load_dotenv()
        return os.getenv(self.api_key_varname)
    

class DocumentPreparer:
    def __init__(self):
        self.separators_for_docs_splitting = ['\n\n', '\n', '.', ',']

    def load_pdf_as_docs(self, pdf_filepath):
        if self.is_path_valid(pdf_filepath):
            loader = PDFMinerLoader(pdf_filepath)
            docs = loader.load()
            return docs
        
        else:
            raise Exception(f"Provided pdf filepath doesn't exist. {pdf_filepath}")
        
    def split_pdf_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter( separators=self.separators_for_docs_splitting )
        docs = splitter.split_documents(docs)
        return docs


        

    def is_path_valid(self, path):
        if os.path.exists(path):
            return True
        else:
            return False