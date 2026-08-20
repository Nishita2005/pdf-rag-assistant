import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from vectorstore import load_vectorstore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load API keys from .env
load_dotenv()

# Verify GROQ_API_KEY is present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

# 2. Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2  # Keep temperature low to prevent hallucination
)

# 3. Custom System Prompt (Grounded Output)
prompt_template = """Use only the following context to answer the user's question. 
If the context does not contain the answer, say "I don't know based on the provided document." 
Do not attempt to fabricate an answer.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# 4. Helper Function to Format Context Documents into Text
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(vectorstore_path="faiss_index"):
    """
    Connects FAISS retriever, custom prompt, Groq LLM, and output parser using LCEL.
    """
    # Load local vector store from Day 3
    vectorstore = load_vectorstore(vectorstore_path)
    
    # Configure retriever to pull top 3 matching chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define LCEL Chain
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


if __name__ == "__main__":
    print("🚀 Initializing Day 4 RAG Pipeline...")
    chain, retriever = build_rag_chain("faiss_index")
    
    print("\n--- RAG Pipeline Ready! Ask a question (type 'exit' to quit) ---\n")
    
    while True:
        query = input("Ask a question: ").strip()
        if not query or query.lower() == "exit":
            print("Exiting RAG CLI...")
            break
            
        print("\n🔎 Searching context and generating response...")
        
        # 1. Inspect retrieved context chunks
        retrieved_docs = retriever.invoke(query)
        print(f"📄 Retrieved {len(retrieved_docs)} chunks from FAISS.")
        
        # 2. Run query through the pipeline
        response = chain.invoke(query)
        
        print("\n🤖 AI Answer:")
        print(response)
        print("-" * 60 + "\n")