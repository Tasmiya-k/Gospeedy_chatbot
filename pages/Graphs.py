import streamlit as st
import pandas as pd
import os
import torch
import shutil
from PIL import Image
from pdf2image import convert_from_bytes
from torchvision import transforms, models
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF
import base64

# Load API key for Gemini
load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Define class labels
CLASS_LABELS = ['just image', 'bar chart', 'diagram', 'flow chart', 'graph', 'growth chart', 'pie chart', 'table']
GRAPH_LABELS = {'bar chart', 'graph', 'growth chart', 'pie chart'}

# Load classification model
def load_model(model_path='chart_classifier_model.pth'):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_LABELS))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model

# Convert PDF pages to images
def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    image_paths = []
    output_folder = "pdf_pages"
    os.makedirs(output_folder, exist_ok=True)
    
    for i, image in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(img_path, "PNG")
        image_paths.append((img_path, i+1))  # Store file path & page number
    return image_paths

# Classify images and filter charts/graphs
def classify_images(image_paths, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    filtered_output_folder = "filtered_graphs"
    os.makedirs(filtered_output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    with torch.no_grad():
        for img_path, page_num in image_paths:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            label = CLASS_LABELS[predicted_class.item()]
            
            if label in GRAPH_LABELS:
                shutil.copy(img_path, os.path.join(filtered_output_folder, os.path.basename(img_path)))
                results.append((img_path, page_num, label))
    return results

# Generate financial insights using Gemini with stronger instructions
def analyze_chart_with_gemini(image_path):
    prompt = """Extract insights from the provided chart/graph image in the following structured format.
    IMPORTANT: Make sure to provide at least 2-3 rows of data for each table.
    
    **Risk Analysis Table:**
    | Risk Category | Summary | Potential Impact (Low/Medium/High) | Likelihood (Low/Medium/High) | Mitigation Strategy |
    |--------------|---------|----------------------------------|------------------------|------------------|
    | [risk category 1] | [summary 1] | [impact 1] | [likelihood 1] | [strategy 1] |
    | [risk category 2] | [summary 2] | [impact 2] | [likelihood 2] | [strategy 2] |
    
    **Positive Indicators Table:**
    | Indicator | Value | Strategic Impact |
    |-----------|-------|------------------|
    | [indicator 1] | [value 1] | [impact 1] |
    | [indicator 2] | [value 2] | [impact 2] |
    
    **Negative Indicators Table:**
    | Indicator | Value | Strategic Impact |
    |-----------|-------|------------------|
    | [indicator 1] | [value 1] | [impact 1] |
    | [indicator 2] | [value 2] | [impact 2] |

    **Instructions:**
    - Identify at least 2-3 potential risks and classify them (e.g., financial, market, operational, regulatory).
    - For each table, ensure you provide at least 2-3 rows of meaningful data.
    - Summarize the key insights from the chart/graph.
    - Assess potential impact and likelihood.
    - Suggest mitigation strategies.
    - Identify positive and negative indicators from the chart/graph with corresponding values and strategic impact.
    - DO NOT leave any table empty.
    - Format properly with | characters separating columns.
    """
    
    try:
        uploaded_file = genai.upload_file(image_path)
        response = model_gemini.generate_content([uploaded_file, prompt])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Improved function to parse tables from the Gemini response
def parse_tables(analysis_text):
    # Default empty tables with placeholders
    default_risk_data = [
        ["Data Unavailable", "Could not extract data", "Medium", "Medium", "Retry analysis"]
    ]
    default_indicator_data = [
        ["Data Unavailable", "N/A", "Could not extract data"]
    ]
    
    risk_df = pd.DataFrame(default_risk_data, columns=["Risk Category", "Summary", "Potential Impact", "Likelihood", "Mitigation Strategy"])
    positive_df = pd.DataFrame(default_indicator_data, columns=["Indicator", "Value", "Strategic Impact"])
    negative_df = pd.DataFrame(default_indicator_data, columns=["Indicator", "Value", "Strategic Impact"])
    
    if "Error" in analysis_text:
        return risk_df, positive_df, negative_df
    
    # Try to extract tables using markdown table parsing
    tables = {}
    current_table = None
    rows = []
    
    for line in analysis_text.split('\n'):
        line = line.strip()
        
        # Detect table headings
        if "**Risk Analysis Table:**" in line:
            current_table = "risk"
            rows = []
        elif "**Positive Indicators Table:**" in line:
            if current_table == "risk" and rows:
                tables["risk"] = rows
            current_table = "positive"
            rows = []
        elif "**Negative Indicators Table:**" in line:
            if current_table == "positive" and rows:
                tables["positive"] = rows
            current_table = "negative"
            rows = []
        # Check for table row (has pipe character)
        elif line and '|' in line:
            # Skip table header separator (e.g., |-----|-----|-----|)
            if '---' not in line:
                cols = [col.strip() for col in line.split('|')[1:-1]]
                if cols and all(col != "" for col in cols):
                    # Skip header rows
                    if cols[0] != "Risk Category" and cols[0] != "Indicator":
                        rows.append(cols)
        # End of the text - save the last table
        elif current_table == "negative" and not line and rows:
            tables["negative"] = rows
    
    # Save the last table if we haven't already
    if current_table and rows and current_table not in tables:
        tables[current_table] = rows
    
    # Create DataFrames from the extracted table data
    if "risk" in tables and tables["risk"]:
        risk_rows = [row for row in tables["risk"] if len(row) == 5]
        if risk_rows:
            risk_df = pd.DataFrame(risk_rows, columns=["Risk Category", "Summary", "Potential Impact", "Likelihood", "Mitigation Strategy"])
    
    if "positive" in tables and tables["positive"]:
        positive_rows = [row for row in tables["positive"] if len(row) == 3]
        if positive_rows:
            positive_df = pd.DataFrame(positive_rows, columns=["Indicator", "Value", "Strategic Impact"])
    
    if "negative" in tables and tables["negative"]:
        negative_rows = [row for row in tables["negative"] if len(row) == 3]
        if negative_rows:
            negative_df = pd.DataFrame(negative_rows, columns=["Indicator", "Value", "Strategic Impact"])
    
    return risk_df, positive_df, negative_df

# Function to generate individual download buttons for CSV
def generate_download_button(df, page_num, table_name):
    # Convert DataFrame to CSV
    csv = df.to_csv(index=False).encode('utf-8')
    
    # Add a unique key to the download button
    st.download_button(
        label=f"Download {table_name} as CSV",
        data=csv,
        file_name=f"{table_name}_page_{page_num}.csv",
        mime='text/csv',
        key=f"download_button_{table_name}_{page_num}"  # Unique key for each download button
    )

# Improved PDF class based on the reference code
class ChartAnalysisPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)
        
    def header(self):
        self.set_font("Arial", 'B', 15)
        self.cell(0, 10, "Chart Analysis Report", ln=True, align='C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def add_title(self, title):
        self.set_font("Arial", 'B', 12)
        self.set_fill_color(230, 230, 250)  # Light lavender background
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(2)
        
    def add_table(self, title, dataframe):
        self.add_title(title)
        
        # Define column widths based on table type
        col_names = list(dataframe.columns)
        
        # Customize column widths based on table type
        if len(col_names) == 5:  # Risk Analysis Table
            col_widths = [30, 45, 30, 30, 45]
        elif len(col_names) == 3:  # Positive/Negative Indicators Tables
            col_widths = [50, 40, 90]
        else:
            # Default: equal width for all columns
            col_widths = [180 / len(col_names)] * len(col_names)
        
        line_height = 6  # Base line height
        
        # Header
        self.set_font("Arial", "B", 10)
        self.set_fill_color(200, 220, 255)  # Light blue header
        for i, col in enumerate(col_names):
            self.cell(col_widths[i], line_height + 2, str(col), border=1, align="C", fill=True)
        self.ln()
        
        # Rows
        self.set_font("Arial", "", 9)
        for _, row in dataframe.iterrows():
            cell_data = [str(row[col]) for col in col_names]
            
            # Step 1: Calculate the height needed for each cell
            cell_lines = []
            for i, cell in enumerate(cell_data):
                # Split cell text to determine number of lines
                # Using a simple approximation based on text length and cell width
                chars_per_line = col_widths[i] / 2  # Approximate chars that fit per line
                num_lines = max(1, len(cell) / chars_per_line) if chars_per_line > 0 else 1
                cell_lines.append(num_lines)
            
            max_lines = max(cell_lines)
            row_height = line_height * max_lines
            
            # Step 2: Check if row will overflow page
            if self.get_y() + row_height > self.page_break_trigger:
                self.add_page()
                
                # Redraw header on new page
                self.set_font("Arial", "B", 10)
                self.set_fill_color(200, 220, 255)
                for i, col in enumerate(col_names):
                    self.cell(col_widths[i], line_height + 2, str(col), border=1, align="C", fill=True)
                self.ln()
                self.set_font("Arial", "", 9)
            
            # Step 3: Draw full-height cells
            y_start = self.get_y()
            x_start = self.l_margin
            
            for i, cell in enumerate(cell_data):
                # Draw cell rectangle
                self.rect(x_start, y_start, col_widths[i], row_height)
                
                # Print cell text within rectangle
                self.set_xy(x_start, y_start)
                self.multi_cell(col_widths[i], line_height, cell, border=0)
                
                # Move to start position for next cell
                x_start += col_widths[i]
            
            # Move to next row
            self.set_y(y_start + row_height)
        
        self.ln(5)  # Add space after table

# Function to create PDF from tables and save to Final_PDF folder
def create_pdf_from_tables(risk_df, positive_df, negative_df, img_path, page_num):
    # Create Final_PDF directory if it doesn't exist
    final_pdf_folder = "Final_PDF"
    os.makedirs(final_pdf_folder, exist_ok=True)
    
    pdf = ChartAnalysisPDF()
    
    # Add chart image to PDF
    img = Image.open(img_path)
    width, height = img.size
    aspect = height / width
    
    # Set a reasonable width for the image
    img_width = 180
    img_height = img_width * aspect
    
    # Add image
    pdf.ln(5)
    pdf.image(img_path, x=15, y=pdf.get_y(), w=img_width, h=img_height)
    pdf.ln(img_height + 10)
    
    # Add tables to PDF
    pdf.add_table("Risk Analysis Table", risk_df)
    pdf.add_table("Positive Indicators Table", positive_df)
    pdf.add_table("Negative Indicators Table", negative_df)
    
    # Save PDF to a file in the Final_PDF folder
    pdf_path = os.path.join(final_pdf_folder, f"analysis_page_{page_num}.pdf")
    pdf.output(pdf_path)
    
    return pdf_path

# Function to create a download link for the PDF
def get_pdf_download_link(pdf_path, page_num):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    return f"""
        <a href="data:application/pdf;base64,{b64_pdf}" download="analysis_page_{page_num}.pdf">
            <button style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;">
                Download Full Analysis as PDF
            </button>
        </a>
    """

# Streamlit App
def main():
    st.title("📊 PDF Chart & Graph Analyzer with AI")
    st.write("Upload a PDF, and the app will extract charts/graphs & analyze them!")
    
    # Create the Final_PDF directory at the beginning
    final_pdf_folder = "Final_PDF"
    os.makedirs(final_pdf_folder, exist_ok=True)
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        st.info("Processing... Please wait!")
        
        # Load model
        model = load_model()
        
        # Convert PDF to images
        images_with_pages = pdf_to_images(uploaded_file.read())
        
        # Classify images
        results = classify_images(images_with_pages, model)
        
        # Display results
        if results:
            st.success(f"Identified {len(results)} charts/graphs!")
            st.info(f"PDF reports are being saved to the '{final_pdf_folder}' folder")
            
            for img_path, page_num, label in results:
                st.subheader(f"Page {page_num}: {label}")
                st.image(img_path, use_container_width=True)
                
                with st.spinner("Analyzing with AI..."):
                    analysis = analyze_chart_with_gemini(img_path)
                    
                    # Convert analysis text to structured table format
                    if analysis.startswith("Error"):
                        st.error(analysis)
                    else:
                        # Parse all three tables from the response
                        risk_df, positive_df, negative_df = parse_tables(analysis)
                        
                        # Display the tables
                        st.write("### AI Analysis:")
                        
                        # Risk Analysis Table
                        st.write("**Risk Analysis**")
                        st.dataframe(risk_df)
                        generate_download_button(risk_df, page_num, "risk_analysis")
                        
                        # Positive Indicators Table
                        st.write("**Positive Indicators**")
                        st.dataframe(positive_df)
                        generate_download_button(positive_df, page_num, "positive_indicators")
                        
                        # Negative Indicators Table
                        st.write("**Negative Indicators**")
                        st.dataframe(negative_df)
                        generate_download_button(negative_df, page_num, "negative_indicators")
                        
                        # Create combined CSV with all tables
                        combined_data = pd.concat([
                            pd.DataFrame([["RISK ANALYSIS TABLE"]], columns=["Table Type"]),
                            risk_df,
                            pd.DataFrame([[""], ["POSITIVE INDICATORS TABLE"]], columns=["Table Type"]),
                            positive_df,
                            pd.DataFrame([[""], ["NEGATIVE INDICATORS TABLE"]], columns=["Table Type"]),
                            negative_df
                        ])
                        
                        combined_csv = combined_data.to_csv(index=False).encode('utf-8')
                        
                        # Add download button for all tables combined as CSV
                        st.download_button(
                            label="Download All Tables as CSV",
                            data=combined_csv,
                            file_name=f"all_tables_page_{page_num}.csv",
                            mime='text/csv',
                            key=f"download_all_{page_num}"
                        )
                        
                        # Create PDF with all tables and image and save to Final_PDF folder
                        with st.spinner("Generating PDF report..."):
                            pdf_path = create_pdf_from_tables(risk_df, positive_df, negative_df, img_path, page_num)
                            st.markdown(get_pdf_download_link(pdf_path, page_num), unsafe_allow_html=True)
                            st.success(f"PDF successfully saved to {pdf_path}")
        else:
            st.warning("No charts or graphs detected!")

if __name__ == "__main__":
    main()