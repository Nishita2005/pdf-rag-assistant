import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(file_path: str):
    """
    Loads a document based on its file extension (.pdf, .txt, .docx).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"📄 Loading document from: {file_path}")
    documents = loader.load()
    print(f"✅ Loaded successfully! Document count/pages: {len(documents)}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Splits loaded documents into smaller text chunks with specified overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks (Chunk Size: {chunk_size}, Overlap: {chunk_overlap})")
    return chunks


if __name__ == "__main__":
    # Test path - point to your sample file in the data folder
    test_file = "data/sample.txt"  # Change to data/sample.pdf if testing PDF
    
    # 1. Load the document
    raw_docs = load_document(test_file)
    
    # 2. Chunk the document
    chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=100)
    
    # 3. Inspect the results
    print("\n" + "="*40)
    print("INSPECTING CHUNK 1")
    print("="*40)
    print("Content:\n", chunks[0].page_content)
    print("\nMetadata:\n", chunks[0].metadata)
    
    if len(chunks) > 1:
        print("\n" + "="*40)
        print("INSPECTING CHUNK 2")
        print("="*40)
        print("Content:\n", chunks[1].page_content)