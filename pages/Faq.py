# Frequently Asked Questions 🤔
import streamlit as st

def display_faq_data(faq_data):
    st.title("Frequently Asked Questions")
    
    # Iterate through each PDF
    for pdf_name, categories in faq_data.items():
        # Display PDF name as main header
        st.markdown(f"## {pdf_name}")
        
        # Iterate through each category for this PDF
        for category, questions_answers in categories.items():
            # Display category as header
            st.markdown(f"### {category}")
            
            # Iterate through the questions and answers in each category
            for question, answer in questions_answers.items():  # Changed from questions_answers to questions_answers.items()
                # Display question
                st.markdown(f"**Q: {question}**")
                # Display answer
                st.write(f"A: {answer}")
                # Add a horizontal line between each question-answer pair
                st.markdown("---")

def main():
    # Initialize session state if it doesn't exist
    if 'faq_data' not in st.session_state:
        st.session_state.faq_data = {}
    
    # Retrieve faq_data from session_state
    faq_data = st.session_state.faq_data
    
    # Check if faq_data exists and is not empty
    if faq_data:
        display_faq_data(faq_data)
    else:
        st.warning("No FAQ data available yet.")
        st.info("Please go to the Chatbot page, upload and process a PDF file to generate FAQ data.")

if __name__ == "__main__":
    main()
