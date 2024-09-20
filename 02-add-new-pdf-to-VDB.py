from langchain_chroma import Chroma
import os
from langchain_codebase.codebase import load_single_pdf, split_docs, load_yaml_file, load_vectorstore, add_docs_to_chroma_vectorstore

print("Running..")
PARAMS_FILEPATH = "src/params.yaml"
params = load_yaml_file(PARAMS_FILEPATH)
vdb_path, chunk_size, chunk_overlap, separators = [params[key] for key in ['database_path', 'chunk_size', 'chunk_overlap', 'revenue_laws_seps']]

def only_keep_source_filename_in_docs_metadata(docs):
    """Keeps only the filename in the docs.metadata['source']."""
    for doc in docs:
        metadata_source = doc.metadata['source']
        metadata_source = os.path.basename(metadata_source)
        doc.metadata['source'] = metadata_source

    return docs
        
    
def main():
    pdf_path, include_images = handle_inputs()
    docs = load_single_pdf(pdf_path, include_images)
    docs = split_docs(docs, chunk_size, chunk_overlap)
    docs = only_keep_source_filename_in_docs_metadata(docs)
    vectorstore = load_vectorstore(vdb_path)

    try:
        add_docs_to_chroma_vectorstore(docs, vectorstore)
        print("Pdf contents added to VDB.")
    except Exception as e:
        print(f"Pdf contents added to VDB. {e}")
        raise

def handle_inputs():
    pdf_path = input("Enter pdf path: ")
    include_images = input("Include images (yes/no): ").strip().lower()

    if not include_images in ['yes', 'no']:
        raise Exception(f"Invalid value for include images. Provided value: {include_images}")
    include_images = (include_images == 'yes')

    return pdf_path, include_images


if __name__ == '__main__':
    main()