import os
import langchain
import faiss
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

print("--- Day 1 Verification ---")
print("✅ LangChain version:", langchain.__version__)
print("✅ Streamlit version:", st.__version__)
print("✅ FAISS initialized successfully!")

# Check for Groq Key
api_key = os.getenv("GROQ_API_KEY")

if api_key and api_key.startswith("gsk_"):
    masked_key = api_key[:7] + "..." + api_key[-4:]
    print(f"✅ Groq API Key detected: {masked_key}")
    print("\n🎉 Environment and key setup completed!")
else:
    print("❌ GROQ_API_KEY is missing or invalid in your .env file.")