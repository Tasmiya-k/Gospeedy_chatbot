# Risk Analysis Chatbot - FIXED VERSION 📃
import numpy as np
import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import requests
from urllib.parse import urlparse
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import joblib
import speech_recognition as sr
import pyttsx3
import threading
import random
from io import BytesIO
from typing import List, Any
from dotenv import load_dotenv

load_dotenv() 

# Set your API key
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# RATE LIMITING IMPLEMENTATION
class GeminiRateLimitHandler:
    def __init__(self, max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 120.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check if it's a rate limit error
                if ("429" in error_str or 
                    "RESOURCE_EXHAUSTED" in error_str or 
                    "quota" in error_str.lower() or
                    "rate limit" in error_str.lower()):
                    
                    if attempt == self.max_retries - 1:
                        st.error(f"Rate limit exceeded after {self.max_retries} attempts. Please upgrade to paid tier or try again later.")
                        raise
                    
                    # Calculate exponential backoff with jitter
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay
                    )
                    
                    st.warning(f"Rate limited. Retrying {attempt + 1}/{self.max_retries} in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    # Non-rate-limit errors should bubble up immediately
                    raise
        
        raise last_exception

# BATCH PROCESSING FOR LARGE DOCUMENTS
class GeminiBatchProcessor:
    def __init__(self, batch_size: int = 3, delay_between_batches: float = 3.0):
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.rate_limiter = GeminiRateLimitHandler()  # Your existing rate limiter

    def process_large_document(self, text_chunks: List[str]) -> FAISS:
        """Process large documents in smaller batches to avoid rate limits"""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_stores = []

        total_batches = (len(text_chunks) + self.batch_size - 1) // self.batch_size
        status_placeholder = st.empty()  # placeholder for dynamic status text

        for batch_index in range(total_batches):
            start = batch_index * self.batch_size
            end = start + self.batch_size
            batch = text_chunks[start:end]

            # Update single line status
            status_placeholder.text(f"Processing batch {batch_index + 1} of {total_batches}...")

            def create_batch_embeddings():
                return FAISS.from_texts(batch, embedding=embeddings)

            batch_vector_store = self.rate_limiter.execute_with_retry(create_batch_embeddings)
            vector_stores.append(batch_vector_store)

            # Wait between batches except after last batch
            if batch_index < total_batches - 1:
                time.sleep(self.delay_between_batches)

        # Clear status text after processing
        status_placeholder.empty()

        # Merge all vector stores
        if len(vector_stores) == 1:
            return vector_stores[0]

        combined_store = vector_stores[0]
        for store in vector_stores[1:]:
            combined_store.merge_from(store)

        return combined_store

# Initialize the batch processor globally
batch_processor = GeminiBatchProcessor(batch_size=3, delay_between_batches=3.0)

# FIXED: Updated embeddings with correct model
def get_embeddings():
    """Get embeddings with rate limiting"""
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Function to get text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to save the vector store to a file
def save_vector_store(vector_store, filename='vector_store.pkl'):
    vector_store.save_local(filename)

# Function to load the vector store from a file
def load_vector_store(filename='vector_store.pkl'):
    if os.path.exists(filename):
        embeddings = get_embeddings()
        return FAISS.load_local(filename, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Failed to load vector store. File does not exist.")
        return None

# FIXED: Updated vector store creation with batch processing and rate limiting
def get_vector_store(text_chunks):
    """Create vector store with batch processing and rate limiting"""
    try:
        # Use batch processing for large documents
        if len(text_chunks) > 5:
            st.info(f"Processing {len(text_chunks)} chunks in batches to avoid rate limits...")
            vector_store = batch_processor.process_large_document(text_chunks)
        else:
            # For small documents, process normally with retry logic
            embeddings = get_embeddings()
            rate_limiter = GeminiRateLimitHandler()
            
            def create_embeddings():
                return FAISS.from_texts(text_chunks, embedding=embeddings)
            
            vector_store = rate_limiter.execute_with_retry(create_embeddings)
        
        # Save vector store to pickle file
        save_vector_store(vector_store)
        st.success("Vector store created and saved successfully!")
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.error("Consider switching to free local embeddings or enabling billing on your Google account.")
        return None

# Function to serialize the FAISS index to bytes
def serialize_to_bytes(vector_store):
    return vector_store.serialize_to_bytes()

# Function to deserialize the FAISS index from bytes
def deserialize_from_bytes(serialized_bytes, embeddings):
    return FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_bytes)

# Function to merge two FAISS vector stores
def merge_vector_stores(vector_store1, vector_store2):
    vector_store1.merge_from(vector_store2)

# FIXED: Updated conversational chain creation
def get_conversational_chain(vector_store, prompt_template):
    """Create conversational chain with proper error handling"""
    if vector_store is None:
        st.error("Vector store is None. Cannot create conversational chain.")
        return None
        
    prompt_template = """
        You are the most accurate model in the world with extensive knowledge of legal contracts and their terms. Answer the question as detailed as possible based on the provided context, ensuring all relevant details are included.

        Instructions:
        - Provide answers only in English.
        - Ensure clear and proper formatting.
        - If the answer is not in the provided context, respond with "Answer is not available in the context."
        - Include units for numerical data (e.g., $, million, billion).
        - If requested, provide the answer in a well-formatted table.
        - For bullet points, ensure each point starts on a new line.
        - For logical reasoning or open-ended questions, provide justified and logically aligned answers.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """
    
    try:
        model_name = "gemini-2.5-flash"
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

# Global variable to store FAQ data
faq_data = {}

# FIXED: Updated PDF processing with better error handling
def process_pdf(pdf_path):
    """Process PDF with improved error handling and rate limiting"""
    prompt_template = ""

    try:
        # Read uploaded file bytes immediately and create an in-memory buffer
        file_bytes = pdf_path.read()
        pdf_buffer = BytesIO(file_bytes)

        # Use pdf_buffer with PyPDF2 for text extraction
        pdf_reader = PdfReader(pdf_buffer)
        main_text = ""
        for page in pdf_reader.pages:
            main_text += page.extract_text()

        if not main_text.strip():
            st.error("No text could be extracted from the PDF.")
            return "Failed to extract text from PDF"

        text_chunks = get_text_chunks(main_text)

        if not text_chunks:
            st.error("No text chunks created from PDF.")
            return "Failed to create text chunks"

        vector_store = get_vector_store(text_chunks)

        if vector_store is None:
            return "Failed to create vector store due to rate limiting. Please try again later or upgrade to paid tier."

        st.session_state.conversation = get_conversational_chain(vector_store, prompt_template)
        st.session_state.faq_conversation = get_conversational_chain(vector_store, prompt_template)

        # Save the uploaded PDF bytes to the "User_Reports" folder
        folder_path = "User_Reports"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.basename(pdf_path.name)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as file:
            file.write(file_bytes)  # Write from stored bytes, NOT pdf_path.read()

        return f"Successfully processed PDF: {file_name}"

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return f"Failed to process PDF: {str(e)}"

# FIXED: Updated FAQ generation with rate limiting
def generate_faqs(pdf_name):
    """Generate FAQs with rate limiting between requests"""
    if st.session_state.faq_conversation is None:
        st.error("FAQ conversation chain not initialized.")
        return
        
    faq_questions = {
        "Liquidity Risk": [
            "What are the company's current liquidity ratios (e.g., current ratio, quick ratio)?",
            "Does the company have sufficient cash flow to cover its short-term obligations?",
            "How efficiently does the company manage its cash conversion cycle, inventory, and receivables?",
        ],
        "Solvency Risk": [
            "What is the company's debt-to-equity ratio, and how has it changed over time?",
            "Is the company at risk of defaulting on its long-term debts?",
            "What is the company's interest coverage ratio (ability to cover interest payments)?",
        ],
        "Market Risk": [
            "How sensitive is the company to changes in interest rates, exchange rates, or commodity prices?",
            "Does the company hedge against foreign currency or commodity price fluctuations?",
            "What external economic factors are most likely to impact the company's financial health?",
        ],
        "Credit Risk": [
            "What is the company's credit rating, and how has it changed over time?",
            "What percentage of the company's receivables is overdue or at risk of default?",
            "How effectively does the company manage credit risk through policies or credit insurance?",
        ]
    }

    global faq_data
    faq_data[pdf_name] = {}

    for category, questions in faq_questions.items():
        pdf_faq = {}
        for question in questions:
            try:
                # Add delay between FAQ questions to avoid rate limiting
                time.sleep(3)
                st.info(f"Generating FAQ for {category}: {question[:50]}...")
                
                response = st.session_state.faq_conversation.invoke({'question': question})
                if response['chat_history']:
                    answer = response['chat_history'][-1].content
                    pdf_faq[question] = answer
                else:
                    pdf_faq[question] = "No answer found"
                    
            except Exception as e:
                st.warning(f"Failed to generate answer for question: {question[:50]}... Error: {str(e)}")
                pdf_faq[question] = f"Error generating answer: {str(e)}"
        
        faq_data[pdf_name][category] = pdf_faq

def user_input(user_q):
    """Handle user input with error checking"""
    if st.session_state.conversation is None:
        st.error("Conversation chain not initialized. Please process a PDF first.")
        return
        
    try:
        response = st.session_state.conversation.invoke({'question': user_q})
        st.session_state.chatHistory = response['chat_history']
        
        # Load vector store for similarity search (optional)
        db = load_vector_store()
        answer = response['answer']
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.append(answer)
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            st.error("Rate limit exceeded. Please wait a moment and try again, or consider upgrading to a paid plan.")

# Function to search for PDFs using Google Custom Search API
def search_for_pdfs(query, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    cse_id = 'f79b6c79519ab4aeb'
    api_key = os.getenv("GOOGLE_API_KEY")
   
    params = {
        'q': f"{query} annual public report filetype:pdf",
        'num': num_results,
        'cx': cse_id,
        'key': api_key,
    }
   
    try:
        response = requests.get(search_url, params=params)
        data = response.json()
        
        pdf_results = []
        if 'items' in data:
            for item in data['items']:
                pdf_url = item['link']
                pdf_name = os.path.basename(urlparse(pdf_url).path)
                pdf_results.append((pdf_name, pdf_url))
        
        return pdf_results
    except Exception as e:
        st.error(f"Error searching for PDFs: {str(e)}")
        return []

def save_processed_data(text_chunks):
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)

def load_processed_data():
    if os.path.exists('processed_data.pkl'):
        with open('processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return None
   
def save_to_csv(df, filename='data.csv'):
    df.to_csv(filename, index=False)

def load_image(image_path, size=(50,50)):
    try:
        image = Image.open(image_path)
        image = image.resize(size)
        return image
    except Exception as e:
        st.warning(f"Could not load image {image_path}: {str(e)}")
        return None

# Load images with error handling
human_image = load_image("human_icon.png", size=(100,100))
chatgpt_image = load_image("bot.png", size=(100,100))

def clear_user_reports_folder():
    folder_path = "User_Reports"
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not remove file {file_path}: {str(e)}")

# Financial terms dictionary (keeping your existing one)
financial_terms = {
    "consideration": "Something of value exchanged for a promise or performance in a contract.",
    "term": "The duration or length of time that a contract is valid.",
    "party": "A person or entity involved in the contract.",
    "indemnity": "A promise to compensate for loss or damage.",
    "breach": "The failure to fulfill the terms of a contract without a legal excuse.",
    "remedy": "The action or procedure that can be taken to enforce a right or redress a wrong in case of a breach.",
    "jurisdiction": "The authority or power of a court to hear and decide a case.",
    "force majeure": "Circumstances beyond the control of the parties that prevent the fulfillment of a contract.",
    "severability": "The ability to separate provisions of a contract that are invalid or unenforceable from those that are valid and enforceable.",
    "confidentiality": "The obligation to keep information shared during the contract period private.",
    "termination": "The ending of a contract before its expiration date.",
    "amendment": "A change or modification to the terms of a contract.",
    "waiver": "The voluntary relinquishment or abandonment of a right or claim under a contract.",
    "governing law": "The laws of a particular jurisdiction that will govern the interpretation and enforcement of the contract.",
    "assignment": "The transfer of rights or obligations under a contract to another party.",
    "representation and warranty": "Statements made by one party in a contract regarding certain facts or circumstances.",
    "dispute resolution": "The process for resolving conflicts or disagreements arising from the contract.",
    "notices": "Formal communications sent between parties to the contract.",
    # Add more terms as needed (keeping your existing dictionary)
}

def underline_financial_terms(response):
    """Highlight financial terms in responses"""
    for term, meaning in financial_terms.items():
        if term in response:
            response = response.replace(term, f'<abbr title="{meaning}">{term}</abbr>')
    return response

# MAIN FUNCTION - UPDATED
def main():
    st.header(" Risk Analysis Chatbot 📃")
    
    # Add rate limit status display
    st.sidebar.subheader("⚡ API Status")
    if st.sidebar.button("Check API Status"):
        try:
            # Test API connection
            embeddings = get_embeddings()
            st.sidebar.success("✅ API connection OK")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.sidebar.error("❌ Rate limit exceeded")
                st.sidebar.info("💡 Consider upgrading to paid tier")
            else:
                st.sidebar.error(f"❌ API Error: {str(e)}")

    # Check if files have been deleted on app start
    if "files_deleted" not in st.session_state:
        st.session_state.files_deleted = False

    # If files haven't been deleted yet, delete them now
    if not st.session_state.files_deleted:
        clear_user_reports_folder()
        st.session_state.files_deleted = True

    # Check session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if 'chatHistory' not in st.session_state:
        st.session_state.chatHistory = None
    if "faq_data" not in st.session_state:
        st.session_state.faq_data = {}

    # Sidebar for settings
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload and Process PDFs")

    # Upload PDFs
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    
    # Process uploaded PDFs
    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs with rate limiting..."):
            for pdf in pdf_docs:
                pdf_name = os.path.basename(pdf.name)
                result = process_pdf(pdf)
                st.write(result)
                
                if "Successfully processed" in result:
                    st.info("Generating FAQs... This may take a few minutes due to rate limiting.")
                    generate_faqs(pdf_name)
                    st.session_state.faq_data = faq_data

    # Display FAQs if available (keeping your existing display logic)
    for pdf_name, categories in st.session_state.faq_data.items():
        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
        st.markdown(f"<h4>{pdf_name}</h4>", unsafe_allow_html=True)
        
        for category, questions_and_answers in categories.items():
            st.markdown(f"<h3>{category}</h3>", unsafe_allow_html=True)
            
            for question, answer in questions_and_answers.items():
                if "Q:" in answer:
                    qa_pairs = answer.split("Q:")
                    for qa_pair in qa_pairs[1:]:
                        q_a_pair = qa_pair.split("A:")
                        if len(q_a_pair) >= 2:
                            question = q_a_pair[0].strip()
                            answer = q_a_pair[1].strip()
                            st.markdown(f'<p style="font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p>A: {answer}</p>', unsafe_allow_html=True)
                else:
                    if "experienced legal assistant" in question or "Tabular format" in question or "|" in answer or "---" in answer or "|---|" in answer:
                        rows = [row.split("|") for row in answer.split("\n") if row.strip()]
                        
                        if rows and len(rows[0]) > 2:
                            rows = [row[1:-1] for row in rows]
                            character = '-'
                            rows = [row for row in rows if not all(cell.strip() == character * len(cell.strip()) for cell in row)]

                            if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                                columns = [col.strip() if col.strip() else f"Column {i+1}" for i, col in enumerate(rows[0])]
                                df = pd.DataFrame(rows[1:], columns=columns)
                                st.write(df.style.set_properties(**{'white-space': 'pre-wrap'}))
                                save_to_csv(df)
                        else:
                            if '<br>' in answer:
                                formatted_answer = answer.replace('<br>', ' ')
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {formatted_answer}</p>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>', unsafe_allow_html=True)

    # Search functionality (keeping your existing search logic)
    company_name = st.sidebar.text_input("Enter Company Name", key="company_name")
    year = st.sidebar.text_input("Enter Year", key="year")
    search_button_clicked = st.sidebar.button("Search")

    if search_button_clicked:
        if company_name and year:
            search_results = search_for_pdfs(company_name + " " + year)
            if search_results:
                st.sidebar.subheader("Search Results:")
                for pdf_name, pdf_url in search_results:
                    st.sidebar.markdown(
                        f'<a href="{pdf_url}" target="_blank" style="display: block; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc; text-decoration: none; color: #333; background-color: #f9f9f9;">{pdf_name}</a>',
                        unsafe_allow_html=True
                    )
            else:
                st.sidebar.write("No results found.")
        else:
            st.sidebar.write("Please enter both company name and year.")

    # Chat history display (keeping your existing display logic but with error handling)
    if st.session_state.chatHistory:
        idx = 0
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                col1, col2 = st.columns([1, 8])
                with col1:
                    if human_image:
                        st.image(human_image, width=40)
                with col2:
                    st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-align: left;">{message.content}</p>', unsafe_allow_html=True)
            else:
                if "|" in message.content and "---" in message.content:
                    rows = [row.split("|") for row in message.content.split("\n") if row.strip()]
                   
                    if rows and len(rows[0]) > 2:
                        rows = [row[1:-1] for row in rows]
                        character = '-'
                        rows = [row for row in rows if not all(cell.strip() == character * len(cell.strip()) for cell in row)]

                        if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                            columns = [col.strip() if col.strip() else f"Column {i+1}" for i, col in enumerate(rows[0])]
                            df = pd.DataFrame(rows[1:], columns=columns)
                            st.write(df)
                            save_to_csv(df)
                            st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
                        else:
                            col1, col2 = st.columns([1, 8])
                            with col1:
                                if chatgpt_image:
                                    st.image(chatgpt_image)
                            with col2:
                                response_with_underline = underline_financial_terms(message.content)
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                            st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
                else:
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        if chatgpt_image:
                            st.image(chatgpt_image)
                    with col2:
                        response_with_underline = underline_financial_terms(message.content)
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                    st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
                   
    # User input
    user_question = st.text_input("Ask a Question from the PDF Files", key="pdf_question")

    if st.button("Get Response") and user_question:
        if st.session_state.conversation is None:
            st.error("Please process a PDF file first before asking questions.")
        else:
            user_input(user_question)

if __name__ == "__main__":
    main()