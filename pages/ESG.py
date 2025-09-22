# --- Final Imports ---
import os
import re
import io
import streamlit as st
import pandas as pd
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import time
import random
from typing import List, Any

# --- RATE LIMITING IMPLEMENTATION ---
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

class GeminiBatchProcessor:
    def __init__(self, batch_size: int = 3, delay_between_batches: float = 3.0):
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.rate_limiter = GeminiRateLimitHandler()
    
    def process_large_document(self, text_chunks: List[str], embeddings) -> FAISS:
        total_batches = (len(text_chunks) + self.batch_size - 1) // self.batch_size
        progress_bar = st.progress(0)
        status_text = st.empty()

        vector_stores = []

        for batch_index in range(total_batches):
            start = batch_index * self.batch_size
            batch = text_chunks[start : start + self.batch_size]

            # Update numeric progress
            status_text.text(f"Processing batch {batch_index + 1} of {total_batches}")
            progress_bar.progress((batch_index + 1) / total_batches)

            def create_batch_embeddings():
                return FAISS.from_texts(batch, embedding=embeddings)

            batch_store = self.rate_limiter.execute_with_retry(create_batch_embeddings)
            vector_stores.append(batch_store)

            if batch_index < total_batches - 1:
                time.sleep(self.delay_between_batches)

        # merge vector stores as before…
        if len(vector_stores) == 1:
            return vector_stores[0]

        combined = vector_stores[0]
        for vs in vector_stores[1:]:
            combined.merge_from(vs)
        return combined

# Initialize global batch processor
batch_processor = GeminiBatchProcessor(batch_size=3, delay_between_batches=3.0)

# --- Functions ---

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages[:2]:  # First 2 pages to detect company name
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_full_text_for_analysis(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def detect_company_name(text, fallback_filename="UnknownCompany"):
    # First, try to find Company: <name> style
    match = re.search(r'Company:\s*(.+)', text, re.IGNORECASE)
    if match:
        company = match.group(1).strip()
    else:
        # Then fallback to something like <Company Name> Annual Report / Sustainability Report
        match2 = re.search(r"([A-Z][A-Za-z0-9&,\.\s\-]{3,})\s+(Annual Report|Sustainability Report|ESG Report|Company|Corporation|Inc\.|Limited|LLC|Ltd\.)", text, re.IGNORECASE)
        if match2:
            company = match2.group(1).strip()
        else:
            company = fallback_filename  # If all fails, use fallback name
    
    # Clean the company name to be safe for filenames
    company = re.sub(r'[^a-zA-Z0-9_]', '_', company)  # Only allow letters, numbers, and underscores
    company = re.sub(r'_+', '_', company)  # Merge multiple underscores
    company = company.strip('_')  # Remove leading/trailing underscores

    return company

# FIXED: Updated ESG analysis with rate limiting
def analyze_esg_risks(text, google_api_key):
    """Analyze ESG risks with rate limiting and batch processing"""
    try:
        # Initialize rate limiter
        rate_limiter = GeminiRateLimitHandler(max_retries=5, base_delay=2.0, max_delay=120.0)
        
        # Create LLM with rate limiting
        def create_llm():
            return ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                model="gemini-1.5-flash-latest",
                temperature=0.0
            )
        
        llm = rate_limiter.execute_with_retry(create_llm)
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        text_chunks = splitter.split_text(text)
        
        st.info(f"Processing {len(text_chunks)} text chunks for ESG analysis...")
        
        # FIXED: Use updated model name and batch processing
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="models/text-embedding-004"  # Updated model name
        )
        
        # Create vector store with batch processing if needed
        if len(text_chunks) > 5:
            st.info("Using batch processing to avoid rate limits...")
            vector_store = batch_processor.process_large_document(text_chunks, embeddings)
        else:
            # For small documents, process normally with retry logic
            def create_embeddings():
                return FAISS.from_texts(text_chunks, embedding=embeddings)
            
            vector_store = rate_limiter.execute_with_retry(create_embeddings)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Define prompt template
        prompt_template = PromptTemplate(
            input_variables=["context"],
            template="""
You are an expert ESG (Environmental, Social, Governance) analyst. Analyze the following company report text carefully.

Focus on extracting meaningful risks and opportunities under the following themes:

Environmental Risks:
- Carbon footprint
- Water usage
- Waste disposal
- Greenhouse gas emissions
- Impact on biodiversity
- Deforestation

Social Risks:
- Wage equality
- Workplace safety and conditions
- Supplier and vendor labor practices
- Human rights violations
- Diversity, equity, and inclusion (DEI)
- Data privacy and security

Governance Risks:
- Transparent communications
- ESG disclosures and reporting
- Board structure, independence, and diversity
- Corruption and fraud prevention
- Organizational integrity and ethics
- Executive compensation practices

---

Return your output STRICTLY in three clearly separated Markdown tables:

### Risk Analysis Table
| Risk Category | Summary of Risk | Potential Impact | Likelihood (Low/Medium/High) | Mitigation Strategy |
|---------------|-----------------|------------------|------------------------------|---------------------|
| ... | ... | ... | ... | ... |

### Positive Indicators Table
| Positive Factor | Current Status | Strategic Impact |
|-----------------|----------------|------------------|
| ... | ... | ... |

### Negative Indicators Table
| Negative Factor | Current Status | Strategic Impact |
|-----------------|----------------|------------------|
| ... | ... | ... |

---

**Important:**
- Only include material (significant) risks and indicators.
- Do not hallucinate any data not present in the text.
- Summarize clearly and professionally.
- Maintain consistency and structure across the tables.

---

Here is the extracted report text:
{context}
"""
        )
        
        # Create QA chain with rate limiting
        def create_qa_chain():
            return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
        
        qa_chain = rate_limiter.execute_with_retry(create_qa_chain)
        
        # Get relevant documents with rate limiting
        def get_relevant_docs():
            return retriever.get_relevant_documents("Extract ESG risks and indicators")
        
        relevant_docs = rate_limiter.execute_with_retry(get_relevant_docs)
        
        # Run the analysis with rate limiting
        def run_analysis():
            return qa_chain.run(input_documents=relevant_docs, question="Extract ESG risks and indicators")
        
        st.info("Running ESG analysis with rate limiting...")
        response = rate_limiter.execute_with_retry(run_analysis)
        
        return response
        
    except Exception as e:
        st.error(f"Error in ESG analysis: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            st.error("Rate limit exceeded. Consider upgrading to a paid Google API plan or try again later.")
        raise

def split_response_to_dfs(response_text):
    def force_pipe_separation(text_block):
        fixed_lines = []
        for line in text_block.splitlines():
            if line.strip() and '|' not in line:
                line = re.sub(r' {2,}', '|', line.strip())
            fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def extract_table(table_text):
        table_text = re.sub(r"\*\*.*?\*\*", "", table_text, flags=re.MULTILINE).strip()
        table_text = force_pipe_separation(table_text)
        table_lines = [line for line in table_text.splitlines() if "|" in line]
        if len(table_lines) > 1:
            cleaned_table = "\n".join(table_lines)
            df = pd.read_csv(io.StringIO(cleaned_table), sep="|", engine="python", skipinitialspace=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df = df.dropna(how="all")
            df = df[~df.isin(['---']).any(axis=1)]
            return df
        else:
            return pd.DataFrame()

    risk_start = response_text.find("Risk Analysis Table")
    positive_start = response_text.find("Positive Indicators Table")
    negative_start = response_text.find("Negative Indicators Table")

    risk_table = response_text[risk_start:positive_start].strip() if positive_start > risk_start else ""
    positive_table = response_text[positive_start:negative_start].strip() if negative_start > positive_start else ""
    negative_table = response_text[negative_start:].strip()

    risks_df = extract_table(risk_table)
    positive_df = extract_table(positive_table)
    negative_df = extract_table(negative_table)

    return risks_df, positive_df, negative_df

def calculate_esg_score(risks_df, positive_df, negative_df):
    score = 50
    explanation = []

    if not positive_df.empty:
        score += len(positive_df) * 2
        explanation.append(f"+{len(positive_df)*2} points for {len(positive_df)} positive ESG indicators.")

    if not negative_df.empty:
        score -= len(negative_df) * 5
        explanation.append(f"-{len(negative_df)*5} points penalty for {len(negative_df)} negative ESG indicators.")

    final_score = max(0, min(100, score))
    explanation.append(f"🌟 Final ESG Score: {final_score}/100.")

    return final_score, explanation

def save_as_pdf(risks_df, positive_df, negative_df, esg_score, explanation, filename: str):
    if not os.path.exists("Final_PDF"):
        os.makedirs("Final_PDF")

    pdf_path = os.path.join("Final_PDF", filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    wrapped_style = ParagraphStyle(
        name='Wrapped',
        parent=normal_style,
        alignment=TA_LEFT,
        fontSize=8,
        leading=10
    )

    def create_wrapped_table(df, title, header_color):
        elements.append(Paragraph(title, styles['Heading2']))
        elements.append(Spacer(1, 8))
        if df.empty:
            elements.append(Paragraph("No data available.", normal_style))
            elements.append(Spacer(1, 12))
            return
        data = [list(df.columns)]
        for row in df.values:
            wrapped_row = []
            for cell in row:
                text = str(cell) if pd.notnull(cell) else ""
                wrapped_row.append(Paragraph(text, wrapped_style))
            data.append(wrapped_row)
        col_count = len(df.columns)
        table = Table(data, colWidths=[None] * col_count, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("Extracted ESG Insights Report", styles['Title']))
    elements.append(Spacer(1, 20))
    create_wrapped_table(risks_df, "Risk Analysis Table", "#003366")
    create_wrapped_table(positive_df, "Positive Indicators Table", "#006400")
    create_wrapped_table(negative_df, "Negative Indicators Table", "#8B0000")

    elements.append(PageBreak())

    elements.append(Paragraph("🌟 Final ESG Score", styles['Heading2']))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"<b>{esg_score} / 100</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("📋 ESG Score Computation Explanation:", styles['Heading2']))
    elements.append(Spacer(1, 10))
    for exp in explanation:
        elements.append(Paragraph(f"• {exp}", normal_style))
        elements.append(Spacer(1, 8))

    doc.build(elements)

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="🌿 ESG Insights Extractor", layout="wide")
    st.title("🌍 ESG Risk Analyzer")
    
    # Add rate limit status display
    st.sidebar.subheader("⚡ API Status")
    if st.sidebar.button("Check API Status"):
        try:
            # Test API connection
            my_api_key = os.getenv("GOOGLE_API_KEY")
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=my_api_key,
                model="models/text-embedding-004"
            )
            st.sidebar.success("✅ API connection OK")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.sidebar.error("❌ Rate limit exceeded")
                st.sidebar.info("💡 Consider upgrading to paid tier")
            else:
                st.sidebar.error(f"❌ API Error: {str(e)}")

    pdf_file = st.file_uploader("📂 Upload Annual Public Report or ESG Report:", type=["pdf"])

    if pdf_file and st.button("🔍 Analyze ESG Insights"):
        with st.spinner("Analyzing ESG Report with rate limiting... please wait 🚀"):
            try:
                report_text = extract_full_text_for_analysis(pdf_file)
                first_page_text = extract_text_from_pdf(pdf_file)

                company_name = detect_company_name(first_page_text, fallback_filename=os.path.splitext(pdf_file.name)[0])
                safe_company_name = company_name

                if not os.path.exists("Final_PDF"):
                    os.makedirs("Final_PDF")

                cache_file_path = f"Final_PDF/cache_response_{safe_company_name}.txt"

                my_api_key = os.getenv("GOOGLE_API_KEY")

                if not os.path.exists(cache_file_path):
                    st.info("Running ESG analysis with rate limiting. This may take a few minutes...")
                    raw_esg_risks = analyze_esg_risks(report_text, my_api_key)
                    with open(cache_file_path, "w", encoding="utf-8") as f:
                        f.write(raw_esg_risks)
                    st.success("ESG analysis completed and cached!")
                else:
                    st.info("Loading cached ESG analysis...")
                    with open(cache_file_path, "r", encoding="utf-8") as f:
                        raw_esg_risks = f.read()

                risks_df, positive_df, negative_df = split_response_to_dfs(raw_esg_risks)

                if not risks_df.empty:
                    st.subheader("Extracted Risk Analysis Table:")
                    st.dataframe(risks_df, use_container_width=True)

                if not positive_df.empty:
                    st.subheader("Extracted Positive Indicators Table:")
                    st.dataframe(positive_df, use_container_width=True)

                if not negative_df.empty:
                    st.subheader("Extracted Negative Indicators Table:")
                    st.dataframe(negative_df, use_container_width=True)

                if risks_df.empty and positive_df.empty and negative_df.empty:
                    st.warning("⚠️ No ESG insights extracted.")
                    return

                esg_score, explanation = calculate_esg_score(risks_df, positive_df, negative_df)

                final_pdf_filename = f"{safe_company_name}_esg_risk.pdf"

                save_as_pdf(risks_df, positive_df, negative_df, esg_score, explanation, filename=final_pdf_filename)
                st.success(f"✅ ESG Insights PDF (Score: {esg_score}/100) saved!")

                with open(os.path.join("Final_PDF", final_pdf_filename), "rb") as f:
                    st.download_button(
                        label="📥 Download ESG Insights Report (PDF)",
                        data=f,
                        file_name=final_pdf_filename,
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"Error during ESG analysis: {str(e)}")
                if "429" in str(e) or "quota" in str(e).lower():
                    st.error("🚨 Rate limit exceeded! Please try one of these solutions:")
                    st.info("1. Wait a few minutes and try again")
                    st.info("2. Enable billing on your Google API account")
                    st.info("3. Use a different API key")
                    st.info("4. Consider switching to free local embedding models")

if __name__ == "__main__":
    main()