import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get text from PDF documents
def get_pdf_text(pdf_docs):
    url='https://www.griet.ac.in/'
    text=""

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='\n')
    except Exception as e:
        print(e)

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load conversational chain
def get_conversational_chain():
    prompt_template = """
**You are now GRIET-GPT created by Baswa Tarun, a Virtual Guide to GRIET! We assist with all queries regarding our college.**

**Understanding and Analyzing the Context:**

Given the comprehensive understanding and analytical capabilities, let's delve deep into the provided context to extract nuanced insights and valuable information.

**Context Overview:**
{context}

**Thorough Question Analysis:**

The questions posed by users serve as catalysts for in-depth exploration and elucidation of the context. Each question warrants meticulous scrutiny and elaborate elucidation.

**User Inquiry:**
"{question}"

**Detailed Response:**

In response to the user inquiry, I'll provide you with a comprehensive and exhaustive explanation rooted in the essence of the provided context. Let's leave no stone unturned in unraveling the intricacies and nuances embedded within the subject matter.

**Response and Analysis:**

If the answer lies within the confines of the provided context, I'll illuminate the path with illuminating insights and precise elucidation. However, if the answer eludes the grasp of the context, I'll gracefully acknowledge the absence of pertinent information, affirming that the "answer is not available in the context."
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    if not user_question.strip():
        st.write("")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    if user_question.strip().lower() in ["hello", "hi", "hey"]:
        # If it's a greeting, provide a custom response and exit the function
      
        st.write("**Reply:**", "Hello! How can I assist you today? Please ask anything about GRIET.")
        return

    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
      
        st.write("**Reply:**", response["output_text"])
    except Exception as e:
      
        st.write("**Reply:**", "Sorry, I couldn't process your request at the moment. Please try again later.")
        print(f"Error: {e}")

# Main function
def main():
    st.set_page_config("GRIET-GPT", page_icon=":robot:")
    st.header(" GRIET-GPT : A Conversational AI for GRIET")



    # Get user input
    user_question = st.text_input("Ask Anything About GRIET:")

    # Handle user input
    
    user_input(user_question)

    pdf_docs = ['Griet.pdf']
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

# Entry point
if __name__ == "__main__":
    main()
