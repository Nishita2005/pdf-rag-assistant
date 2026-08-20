import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from chunking import load_document, chunk_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Initialize local embedding model (Runs on CPU, completely free)
print("⏳ Loading embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
print("✅ Embedding model ready.")


def create_and_save_vectorstore(chunks, store_path="faiss_index"):
    """
    Generates embeddings for chunks and saves the FAISS index to disk.
    """
    print(f"⚙️ Generating embeddings and indexing {len(chunks)} chunks into FAISS...")
    
    # Generate embeddings and build vector database
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Save the vector database locally
    vectorstore.save_local(store_path)
    print(f"💾 FAISS index saved locally at: '{store_path}'")
    return vectorstore


def load_vectorstore(store_path="faiss_index"):
    """
    Loads an existing FAISS index from disk.
    """
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Vector store path '{store_path}' not found.")
        
    print(f"📂 Loading existing FAISS index from: '{store_path}'")
    vectorstore = FAISS.load_local(
        folder_path=store_path, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Required for loading local pickle index files
    )
    print("✅ FAISS index loaded into memory.")
    return vectorstore


if __name__ == "__main__":
    # Test file path inside your data folder
    test_file = "data/sample.txt"
    index_folder = "faiss_index"

    # Step A: Load, Chunk, and Index Document
    raw_docs = load_document(test_file)
    chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=100)
    
    # Create and save FAISS index
    vectorstore = create_and_save_vectorstore(chunks, store_path=index_folder)

    # Step B: Test Similarity Search
    query = "What is Retrieval-Augmented Generation?"
    print(f"\n🔍 Testing Similarity Search for query: '{query}'")
    
    # Retrieve top 2 most semantically relevant chunks
    results = vectorstore.similarity_search(query, k=2)

    print("\n" + "="*50)
    print("TOP RETRIEVED CHUNK:")
    print("="*50)
    print(results[0].page_content)
    print("\nSource Metadata:", results[0].metadata)