#Latest News and Stock Data 📰
from fpdf import FPDF
import streamlit as st
from newsapi import NewsApiClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
import yfinance as yf
import plotly.graph_objs as go
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import pandas as pd


import nltk

# 1. Download the standard punkt tokenizer if not already present
nltk.download('punkt', quiet=True)

# 2. (Optional) Download stopwords if not already present
nltk.download('stopwords', quiet=True)

# 3. If your environment truly requires a custom 'punkt_tab', handle gracefully:
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # Fallback to the standard 'punkt' tokenizer
    nltk.download('punkt')

# Now import tokenizers and other modules
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


# Initialize News API client
newsapi = NewsApiClient(api_key='1dbf0f8c992e4377ae74790b6cfc0d3c')
# GET FROM https://newsapi.org/

# Function to fetch latest news about the company
def get_latest_news(company):
    # Search for news articles related to the company
    news = newsapi.get_everything(q=company, language='en', sort_by='publishedAt', page_size=10)
    # Filter articles that explicitly mention the company name
    news['articles'] = [
        article for article in news['articles']
        if company.lower() in (article['title']+" "+article['description']+" "+article.get('content','')).lower()
    ]
    return news['articles']

# Function to preprocess and summarize news articles
def summarize_news(article):
    # Check if article content is available
    if 'content' in article and article['content']:
        text = article['content']
        # Remove timestamp
        text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\b', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        stopwords_list = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stopwords_list]
        freq_dist = FreqDist(words)
        most_common_words = freq_dist.most_common(10)
        # Identify relevant sentences
        relevant_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word, _ in most_common_words):
                relevant_sentences.append(sentence)
        # Join relevant sentences to form summary
        summary = ' '.join(relevant_sentences)
        return summary
    else:
        return None

# Function to fetch the full text of an article
def fetch_full_article(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract the main content of the article
            paragraphs = soup.find_all('p')
            article_text = '\n'.join([para.get_text() for para in paragraphs])
            return article_text  # Return the cleaned text content
        else:
            return None
    except Exception as e:
        print(f"Error fetching article: {e}")
        return None


# Function to create a PDF from news articles
def create_pdf(articles, company_name):
    pdf_path = f"news_pdfs/{company_name.replace(' ', '_')}_news.pdf"
    
    # Delete old PDFs
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    # Create a new PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"News Articles for {company_name}", styles['Title']))
    story.append(Spacer(1, 12))

    for article in articles:
        if 'url' in article:
            full_text = fetch_full_article(article['url'])
            if full_text:
                story.append(Paragraph(article['title'], styles['Heading2']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(full_text, styles['BodyText']))
                story.append(Spacer(1, 12))  # Add space between articles

    doc.build(story)
    return pdf_path
from langchain.chains import LLMChain

# analyze_pdf_with_langchain
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter


def analyze_pdf_with_langchain(pdf_path):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Extract text from all pages
    full_text = "\n\n".join([doc.page_content for doc in documents])

    # Load your Google API key from environment variable
    google_api_key = os.getenv("GOOGLE_API_KEY")
    # Initialize LLM model
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    # **Updated Risk Analysis Prompt** (Forces Structured Table Output)
    risk_report_prompt = """
    You are an AI risk analyst. Analyze the following news articles for all possible risks related to the company mentioned.
    - Identify risk categories: Financial, Legal, Operational, Reputational, Ethical, Cybersecurity, Regulatory, etc.
    - Provide a short summary for each risk.
    - Mention the potential impact (Low, Medium, High).
    - Assess the likelihood of occurrence (Low, Medium, High).
    - Suggest risk mitigation measures.

    **Output Format:**  
    Return data in **CSV format** with the following columns:  
    **Risk Category, Summary, Potential Impact, Likelihood, Mitigation Strategy**  
    Separate columns using `|` (pipe symbol). Do not include any extra text.

    Document: {text}
    """

    # Create a prompt template
    prompt = PromptTemplate(template=risk_report_prompt, input_variables=["text"])

    # Generate risk analysis
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    risk_analysis_output = llm_chain.run({"text": full_text})

    # **Parsing the Output**
    risk_data = []
    for row in risk_analysis_output.strip().split("\n"):
        columns = row.split("|")
        if len(columns) == 5:  # Ensuring correct structure
            risk_data.append({
                "Risk Category": columns[0].strip(),
                "Summary": columns[1].strip(),
                "Potential Impact": columns[2].strip(),
                "Likelihood": columns[3].strip(),
                "Mitigation Strategy": columns[4].strip(),
            })

    # **Handle Edge Case: If No Data is Parsed**
    if not risk_data:
        risk_data = [{"Risk Category": "No Data", "Summary": "LLM might not have formatted output correctly", "Potential Impact": "-", "Likelihood": "-", "Mitigation Strategy": "-"}]

    # Convert to DataFrame
    df = pd.DataFrame(risk_data)

    # Save as CSV
    csv_path = pdf_path.replace(".pdf", "_risk_analysis.csv")
    df.to_csv(csv_path, index=False)
    
    strength_matrix_prompt = """
    You are a strategic business analyst. Analyze the following document for strengths and opportunities of the company mentioned.
    - Focus on aspects such as Financial Performance, Innovation, Market Position, ESG, Operational Excellence, etc.
    - Identify positive indicators from the document.
    - Mention the strategic impact of each indicator.

    **Output Format:**  
    Return data in **CSV format** with the following columns:  
    **Category, Positive Indicator, Strategic Impact**  
    Separate columns using `|` (pipe symbol). Do not include any extra text.

    Document: {text}
    """

    # Run Strength & Opportunity Matrix
    strength_prompt = PromptTemplate(template=strength_matrix_prompt, input_variables=["text"])
    strength_chain = LLMChain(llm=llm, prompt=strength_prompt)
    strength_output = strength_chain.run({"text": full_text})

    # Parse strength matrix output
    strength_data = []
    for row in strength_output.strip().split("\n"):
        columns = row.split("|")
        if len(columns) == 3:
            strength_data.append({
                "Category": columns[0].strip(),
                "Positive Indicator": columns[1].strip(),
                "Strategic Impact": columns[2].strip(),
            })

    # Fallback if no data
    if not strength_data:
        strength_data = [{"Category": "No Data", "Positive Indicator": "LLM might not have formatted output correctly", "Strategic Impact": "-"}]

    # Convert to DataFrame
    strength_df = pd.DataFrame(strength_data)

    # Save to CSV if needed
    strength_csv_path = pdf_path.replace(".pdf", "_strength_matrix.csv")
    strength_df.to_csv(strength_csv_path, index=False)

    return df, csv_path, strength_df, strength_csv_path

from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Risk & Opportunity Report", ln=True, align="C")
        self.ln(10)

    def add_table(self, title, dataframe):
        self.set_font("Arial", "B", 12)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(2)

        col_names = list(dataframe.columns)
        col_widths = [190 // len(col_names)] * len(col_names)
        line_height = 6

        # Header
        self.set_font("Arial", "B", 10)
        for i, col in enumerate(col_names):
            self.cell(col_widths[i], line_height + 2, str(col), border=1, align="C", fill=True)
        self.ln()

        # Rows
        self.set_font("Arial", "", 9)
        for _, row in dataframe.iterrows():
            cell_data = [str(row[col]) for col in col_names]

            # Step 1: Calculate the height needed for each cell
            cell_lines = [
                self.multi_cell(col_widths[i], line_height, cell, split_only=True)
                for i, cell in enumerate(cell_data)
            ]
            max_lines = max(len(lines) for lines in cell_lines)
            row_height = line_height * max_lines

            # Step 2: Check if row will overflow page
            if self.get_y() + row_height > self.page_break_trigger:
                self.add_page()

                # Redraw header on new page
                self.set_font("Arial", "B", 10)
                for i, col in enumerate(col_names):
                    self.cell(col_widths[i], line_height + 2, str(col), border=1, align="C", fill=True)
                self.ln()
                self.set_font("Arial", "", 9)

            # Step 3: Draw full-height cells
            y_start = self.get_y()
            for i, cell in enumerate(cell_data):
                x_start = self.get_x()
                self.rect(x_start, y_start, col_widths[i], row_height)
                self.multi_cell(col_widths[i], line_height, cell, border=0, align="L")
                self.set_xy(x_start + col_widths[i], y_start)
            self.ln(row_height)

        self.ln(5)



def save_combined_pdf(risk_df, strength_df, company_name):
    if not os.path.exists("Final_PDF"):
        os.makedirs("Final_PDF")

    pdf = PDFReport()
    pdf.add_page()
    pdf.add_table("Risk Analysis Table", risk_df)
    pdf.add_table("Strength & Opportunity Matrix", strength_df)

    pdf_path = f"Final_PDF/{company_name.replace(' ', '_')}_Report.pdf"
    pdf.output(pdf_path)

    return pdf_path




# Function to fetch stock data
def get_stock_data(symbol):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.Ticker(symbol)
    # Get historical market data
    history = stock_data.history(period="1d", interval="1m")
    return history

# Streamlit app
def News():
    st.title("Latest News Updates 🗞️ and Stock Data 📈")

    # Get user input for company name
    company_name = st.text_input("Enter a company name:", key="company_name_input")

    # Fetch and display news
    if company_name:
        news = get_latest_news(company_name)
        pdf_path = create_pdf(news, company_name)
        for article in news:
            summary = summarize_news(article)
            if summary:
                st.markdown(
                    f"""
                    <div style='position: relative; border-radius: 15px; overflow: hidden; height: 340px; margin-bottom: 20px'>
                        <img src="{article['urlToImage']}" style='width: 100%; height: 100%; object-fit: cover; filter: brightness(30%);'>
                        <div style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(255, 255, 255, 0.3);'></div>
                        <div style='position: absolute; top: 20px; left: 20px; right: 20px; bottom: 20px; color: white;'>
                            <h3 style='color: white;'>{article['title']}</h3>
                            <p><strong>Source:</strong> {article['source']['name']}</p>
                            <p>{summary}</p>
                            <a href="{article['url']}" target="_blank" style='color: white;'>Read more</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='position: relative; border-radius: 15px; overflow: hidden; height: 340px;'>
                        <img src="{article['urlToImage']}" style='width: 100%; height: 100%; object-fit: cover; filter: brightness(30%);'>
                        <div style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(255, 255, 255, 0.3);'></div>
                        <div style='position: absolute; top: 20px; left: 20px; right: 20px; bottom: 20px; color: white;'>
                            <h3 style='color: white;'>{article['title']}</h3>
                            <p><strong>Source:</strong> {article['source']['name']}</p>
                            <p>Summary not available.</p>
                            <a href="{article['url']}" target="_blank" style='color: white;'>Read more</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        # Analyze the PDF and display the report
        
        # risk_df, risk_path, strength_df, strength_path, pdf_path = analyze_pdf_with_langchain(f"news_pdfs/{company_name}_news.pdf")
        risk_df, risk_path, strength_df, strength_path = analyze_pdf_with_langchain(f"news_pdfs/{company_name}_news.pdf")
        pdf_path = save_combined_pdf(risk_df, strength_df, company_name)


        st.markdown("## Risk Analysis Table")
        st.dataframe(risk_df, use_container_width=True)

        st.markdown("## Strength and Opportunity Matrix")
        st.dataframe(strength_df, use_container_width=True)


        # Show the download button
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Final Report (PDF)",
                data=f,
                file_name=f"{company_name}_Risk_Opportunity_Report.pdf",
                mime="application/pdf"
            )


        
        
        
    stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL):", key="stock_symbol_input")
    # Fetch and display stock data
    if stock_symbol:
        stock_data = get_stock_data(stock_symbol)
        if not stock_data.empty:
            # Display current stock price
            current_price = stock_data['Close'].iloc[-1]
            # Display current stock price with custom styling
            st.markdown(
                f"""
                <div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">
                    <h3 style="color:#333333;text-align:center;">Current Stock Price</h3>
                    <h2 style="color:#007bff;font-weight:bold;text-align:center;margin-top:10px;">${current_price:.2f}</h2>
                    <p style="color:#666666;text-align:center;margin-top:10px;">(Last Updated: {stock_data.index[-1]})</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.subheader(f"Stock Data for {stock_symbol}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title="Closing Price Over Time", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)
            st.write("The graph above shows the closing price of the stock over time. Use the zoom and pan tools to explore the data.")
            
            # Plot high price over time
            fig_high = go.Figure()
            fig_high.add_trace(go.Scatter(x=stock_data.index, y=stock_data['High'], mode='lines', name='High Price'))
            fig_high.update_layout(title="High Price Over Time", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_high)
            st.write("The graph above shows the highest price of the stock reached each day over time. Use the zoom and pan tools to explore the data.")
            
            # Plot low price over time
            fig_low = go.Figure()
            fig_low.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Low'], mode='lines', name='Low Price'))
            fig_low.update_layout(title="Low Price Over Time", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_low)
            st.write("The graph above shows the lowest price of the stock reached each day over time. Use the zoom and pan tools to explore the data.")

        else:
            st.warning(f"No stock data found for symbol: {stock_symbol}")

# Run the app
if __name__ == "__main__":
    News()
