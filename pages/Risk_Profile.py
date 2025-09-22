import streamlit as st
import os
from PyPDF2 import PdfMerger, PdfReader
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
import re

# ---------- UTILS ----------

def merge_pdfs(input_folder, output_path):
    merger = PdfMerger()
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".pdf"):
            merger.append(os.path.join(input_folder, filename))
    merger.write(output_path)
    merger.close()

def parse_merged_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Parse Risk Analysis
    high_impact = len(re.findall(r"\bHigh\b", full_text))
    medium_impact = len(re.findall(r"\bMedium\b", full_text))
    low_impact = len(re.findall(r"\bLow\b", full_text))

    total_risks = high_impact + medium_impact + low_impact

    # Parse Positive Indicators
    positive_indicators = len(re.findall(r"Positive Indicator", full_text))

    # Parse ESG Score
    esg_score_match = re.search(r"Final ESG Score\s*[:\-]?\s*(\d+)\s*/\s*100", full_text)
    esg_score = int(esg_score_match.group(1)) if esg_score_match else 70

    # Parse ESG Explainability
    esg_explainability = ""
    explainability_match = re.search(r"ESG Score Explainability:(.*?)Final ESG Score", full_text, re.DOTALL)
    if explainability_match:
        esg_explainability = explainability_match.group(1).strip()

    # Parse Risk Categories
    categories = {}
    matches = re.findall(r"(Financial|Legal|Operational|Reputational|Cybersecurity|Regulatory|Market)\s+Risk", full_text)
    for match in matches:
        categories[match] = categories.get(match, 0) + 1

    return {
        "total_risks": total_risks,
        "high_impact": high_impact,
        "medium_impact": medium_impact,
        "low_impact": low_impact,
        "positive_indicators": positive_indicators,
        "categories": categories,
        "esg_score": esg_score,
        "esg_explainability": esg_explainability
    }

def generate_pie_chart(high, medium, low, filename):
    labels = 'High Impact', 'Medium Impact', 'Low Impact'
    sizes = [high, medium, low]
    colors = ['red', 'orange', 'yellow']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

def generate_bar_graph(categories, filename):
    names = list(categories.keys())
    values = list(categories.values())
    fig, ax = plt.subplots()
    ax.bar(names, values, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_gauge_meter(esg_score, filename):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.barh(0, esg_score, color="green")
    ax.barh(0, 100 - esg_score, left=esg_score, color="lightgray")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title(f"ESG Score: {esg_score}/100")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---------- STREAMLIT MAIN APP ----------

def main():
    st.set_page_config(page_title="Risk Profile Generator", layout="wide")
    st.title("📄 Company Risk Profile Generator (No LLM, Fast & Reliable)")

    if "risk_data" not in st.session_state:
        st.session_state["risk_data"] = None
    if "merged_pdf_path" not in st.session_state:
        st.session_state["merged_pdf_path"] = None
    if "risk_profile_path" not in st.session_state:
        st.session_state["risk_profile_path"] = None

    if st.button("🚀 Merge PDFs and Generate Reports"):
        if not os.path.exists("Merged_PDF"):
            os.makedirs("Merged_PDF")
        merged_pdf_path = "Merged_PDF/Consolidated_Report.pdf"
        merge_pdfs("Final_PDF", merged_pdf_path)

        risk_data = parse_merged_pdf(merged_pdf_path)

        st.session_state["risk_data"] = risk_data
        st.session_state["merged_pdf_path"] = merged_pdf_path

        st.success("Merged and Parsed PDFs Successfully!")

    if st.session_state["risk_data"]:
        risk_data = st.session_state["risk_data"]

        st.header("📈 Risk Profile Summary")

        # --- Overall Risk Score Calculation ---
        total_risks = risk_data['total_risks']
        high = risk_data['high_impact']
        medium = risk_data['medium_impact']
        low = risk_data['low_impact']
        esg_score = risk_data['esg_score']

        raw_score = (3 * high + 2 * medium + 1 * low) / (total_risks if total_risks else 1)
        adjusted_score = raw_score * (1 + (1 - esg_score/100))
        final_score = round(adjusted_score, 2)

        st.subheader(f"🎯 Overall Risk Score: {final_score}/10")
        st.markdown("**(Higher score = Higher risk exposure)**")
        st.markdown("""
        **Calculation Formula:**  
        - Risk Score = (3 × High + 2 × Medium + 1 × Low) / Total Risks  
        - Adjusted Risk Score = Risk Score × (1 + (1 - ESG Score / 100))
        """)

        # --- Key Risk Table ---
        st.subheader("📋 Key Risk Summary")
        summary_data = {
            "Metric": ["Total Risks", "High Impact Risks", "Medium Impact Risks", "Low Impact Risks", "Positive Indicators"],
            "Count": [total_risks, high, medium, low, risk_data['positive_indicators']]
        }
        st.table(pd.DataFrame(summary_data))

        # --- Pie Chart and Bar Graph ---
        st.subheader("📊 Risk Distribution by Impact")
        pie_path = "pie_chart.png"
        bar_path = "bar_graph.png"
        gauge_path = "gauge.png"

        generate_pie_chart(high, medium, low, pie_path)
        generate_bar_graph(risk_data["categories"], bar_path)
        generate_gauge_meter(esg_score, gauge_path)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pie_path, caption="Risk Distribution (Pie Chart)")
        with col2:
            st.image(bar_path, caption="Risks by Category (Bar Chart)")

        # --- ESG Score ---
        st.subheader("🌱 ESG Score Snapshot")
        st.metric(label="Final ESG Score", value=f"{esg_score}/100")
        st.image(gauge_path, caption="ESG Score Gauge Meter")

        st.markdown("**Explainability from Merged Report:**")
        st.markdown(f"> {risk_data['esg_explainability']}")

        # --- Generate Final Risk Profile PDF (if not done already) ---
        risk_profile_path = "Merged_PDF/Risk_Profile.pdf"
        if not os.path.exists(risk_profile_path):
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Company Risk Profile", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, f"Overall Risk Score: {final_score}/10\n(Higher is worse)")
            pdf.ln(5)
            pdf.multi_cell(0, 8, "Risk Score = (3×High + 2×Medium + 1×Low) / Total\nAdjusted for ESG Score penalty")
            pdf.output(risk_profile_path)

        st.session_state["risk_profile_path"] = risk_profile_path

        # --- Download buttons ---
        st.subheader("📥 Download Reports")
        with open(st.session_state["merged_pdf_path"], "rb") as f:
            st.download_button("Download Consolidated Report", f, file_name="Consolidated_Report.pdf", mime="application/pdf")
        with open(st.session_state["risk_profile_path"], "rb") as f:
            st.download_button("Download Risk Profile", f, file_name="Risk_Profile.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
