from src.langchain_codebase import create_rag_chain_from_pdf_path_using_google_and_chroma, chat_loop, load_yaml_file


pdf_filepath = input("Enter the pdf path: ")
params = load_yaml_file("params.yaml")
rag_chain = create_rag_chain_from_pdf_path_using_google_and_chroma(pdf_filepath, params["revenue_laws_seps"], params["chunk_size"], params["chunk_overlap"], params["database_name"], params["system_role"])
chat_loop(rag_chain)