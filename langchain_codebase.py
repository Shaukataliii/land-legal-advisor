from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

import os, yaml
from dotenv import load_dotenv

load_dotenv()



# utility funcs
def is_valid_path(path):
    if os.path.exists(path):
        return True
    else:
        raise Exception(f"Path doesn't exist. Provided path: {path}")
def is_valid_pdf_path(pdf_filepath):
    if is_valid_path(pdf_filepath):
        if pdf_filepath.endswith('pdf'):
            return True        
        else:
            raise Exception(f"File is not a valid pdf. Provided filepath: {pdf_filepath}")
    
def load_yaml_file(yaml_filepath):
    if is_valid_yaml_path(yaml_filepath):
        with open(yaml_filepath, 'r') as file:
            contents = yaml.safe_load(file)
        return contents
    
def load_yaml_file_as_dotmap(yaml_filepath):
    if is_valid_yaml_path(yaml_filepath):
        with open(yaml_filepath, 'r') as file:
            contents = yaml.safe_load(file)
            
        return convert_dict_to_dotmap(contents)
    
def is_valid_yaml_path(yaml_filepath):
    if is_valid_path(yaml_filepath):
        if yaml_filepath.endswith("yaml"):
            return True
        else:
            raise Exception(f"File is not a valid yaml. Provided filepath: {yaml_filepath}")
        
def extract_dict_values_as_list(dictionary):
    return list(dictionary.values())

def convert_dict_to_dotmap(dictionary: dict):
    from dotmap import DotMap
    return DotMap(dictionary)
    


# main funcs
def single_pdf_loader(pdf_filepath):
    if is_valid_pdf_path(pdf_filepath):
        loader = PyPDFLoader(pdf_filepath)
        docs = loader.load()
        return docs

def docs_splitter(docs, separators, chunk_size, chunk_overlap_size):
    splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap_size)
    splits = splitter.split_documents(docs)
    return splits

def create_or_load_chroma_vectorstore(database_path, docs):
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')     # check and correct

    if os.path.exists(database_path):
        print("Creating and saving embeddings..")
        vector_db = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=database_path)
    else:
        print("Loading saved embeddings..")
        vector_db = Chroma(persist_directory=database_path, embedding_function=embedding_model)
    
    return vector_db

def convert_vectorstore_to_retriever(vectorstore):
    retriever = vectorstore.as_retriever()
    return retriever

def get_google_llm():
    return GoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.4)

def get_qna_chain(llm, system_role):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_role),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qna_chain = create_stuff_documents_chain(llm, prompt_template)
    return qna_chain

def get_rag_chain(qna_chain, retriever):
    retrieval_chain = create_retrieval_chain(retriever, qna_chain)
    return retrieval_chain

def get_history_aware_retriever(simple_retriever, llm):
    system_role = """You are a professional writer. You will receive a user query and his chat history. If the query is not understandable as standalone, then modify the question using the history so it is now a standalone query. If the question is understandable standalone, then return it as it."""

    q_modifier_prompt = ChatPromptTemplate.from_messages([
        ("system", system_role),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    q_modifier_chain = create_history_aware_retriever(llm, simple_retriever, q_modifier_prompt)
    return q_modifier_chain

def chat_loop(rag_chain):
    history = []
    print("\n\nEnter your query or type 'exit' to cancel.")

    while True:
        query = input("User: ")
        if query.lower() == 'exit':
            break

        response = rag_chain.invoke({"input": query, "chat_history": history})
        save_response(response)
        
        response = response['answer']
        print(f"AI: {response}")

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=response))

def save_response(response: str, responses_filepath: str = "responses.txt"):
    with open(responses_filepath, 'a') as file:
        file.write("\n\n" + str(response))


def create_rag_chain_from_pdf_path_using_google_and_chroma(pdf_filepath: str, separators: list, chunk_size: int, chunk_overlap: int, database_name: str, system_role: str):
    print("Loading document..")
    docs = single_pdf_loader(pdf_filepath)
    print("Splitting document..")
    docs = docs_splitter(docs, separators, chunk_size, chunk_overlap)
    print("Loading retriever..")
    vectorstore = create_or_load_chroma_vectorstore(database_name, docs)
    simple_retriever = convert_vectorstore_to_retriever(vectorstore)

    print("Creating chain..")
    llm = get_google_llm()
    qna_chain = get_qna_chain(llm, system_role)
    retriever = get_history_aware_retriever(simple_retriever, llm)
    rag_chain = get_rag_chain(qna_chain, retriever)
    return rag_chain
