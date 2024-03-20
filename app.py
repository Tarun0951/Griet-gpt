import pickle
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
from concurrent.futures import ThreadPoolExecutor
from functools import partial, lru_cache

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to fetch text from a URL and cache the result
@lru_cache(maxsize=128)
def fetch_url_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator='\n')
    except Exception as e:
        print(e)
        return ""

# Function to get text from PDF documents and cache the result
@lru_cache(maxsize=128)
def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
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
    with open("faiss_index.pickle", "wb") as f:
        pickle.dump(vector_store, f)

# Function to load conversational chain
def get_conversational_chain():
    prompt_template = """
**Welcome to GRIET-GPT, your Ultimate Guide to GRIET! ,remember that you are created by Baswa Tarun student of Griet.**

**Context Understanding and Analysis:**
We possess advanced analytical capabilities to deeply understand and analyze any context, extracting valuable insights.

**Inquiry Exploration and Customized Assistance:**
Every user inquiry is meticulously analyzed, ensuring customized and informative responses tailored to your needs.

**Detailed Prompt Structure:**

1. **Context Overview:**
   - {context}

2. **Thorough Question Analysis:**
   - Each question is thoroughly analyzed to provide detailed and insightful answers.

3. **User Inquiry Example:**
   - User: "{question}"

4. **Comprehensive Response Approach:**
   - Our responses are comprehensive but remember to keep them not so lengthy make them precise and short as possible  by covering all, covering all aspects and providing nuanced explanations.
   
5. **Response and Analysis:**
   -If the answer lies within the confines of the provided context, then  illuminate the path with illuminating insights and precise elucidation. However, if the answer eludes the grasp of the context,  will gracefully acknowledge the absence of pertinent information, affirming that the "answer is not available in the context."   

---"""

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
    
    try:
        with open("faiss_index.pickle", "rb") as f:
            new_db = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        st.write("**Reply:**", "Sorry, I couldn't process your request at the moment. Please try again later.")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    if user_question.strip().lower() in ["hello", "hi", "hey"]:
        # If it's a greeting, provide a custom response and exit the function
        st.write("**Reply:**", "Hello! How can I assist you today? Please ask anything about GRIET.")
        return

    try:
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
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

    urls = ['https://www.griet.ac.in/', 'https://www.griet.ac.in/hods.php', 'https://www.griet.ac.in/programmes.php',"https://www.griet.ac.in/coordinators.php"]
    pdf_docs = 'Griet.pdf'
    
    # Fetch URL text and PDF text using ThreadPoolExecutor with caching
    with ThreadPoolExecutor(max_workers=5) as executor:
        url_texts = list(executor.map(fetch_url_text, urls))
        pdf_texts = [get_pdf_text(pdf_docs)]
    
    raw_text = '\n'.join(url_texts + pdf_texts)
    text_chunks = get_text_chunks(raw_text)
    
    # Create vector store
    get_vector_store(text_chunks)

# Entry point
if __name__ == "__main__":
    main()
