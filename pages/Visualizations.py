# Visualize the Reports 📊
import streamlit as st
import os
import PyPDF2
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import plotly.express as px

# Function to extract text from PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        # Extract text from each page
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to generate word cloud
def generate_wordcloud(text):
    custom_stopwords = ['company', 'companies', 'business', 'report', 'annual', 'quarterly', 
    'financial', 'statements', 'statement', 'performance', 'results', 
    'result', 'year', 'years', 'quarter', 'quarters', 'data', 'information', 
    'analysis', 'review', 'management', 'overview', 'highlight', 'highlights', 
    'review', 'reviewed', 'include', 'includes', 'including', 'discussed', 
    'discusses', 'discussing', 'highlighted', 'present', 'presents', 
    'presented', 'focus', 'focused', 'focusing', 'focuses', 'significant', 
    'important', 'key', 'importantly', 'note', 'notes', 'noted', 'point', 
    'points', 'pointed', 'address', 'addresses', 'addressed', 'addressing', 
    'highlight', 'highlights', 'highlighted', 'highlighting', 'discuss', 
    'discusses', 'discussed', 'discussing', 'overview', 'overviews', 
    'overviewed', 'overviewing', 'analyze', 'analyzes', 'analyzed', 
    'analyzing', 'analyze', 'analyzes', 'analyzed', 'analyzing', 
    'analytic', 'analytics', 'analytical', 'analyses', 'analytics', 
    'analysis', 'analyses', 'analyst', 'analysts', 'analyzed', 
    'analyzing', 'shareholder', 'shareholders', 'stakeholder', 
    'stakeholders', 'investor', 'investors', 'investing', 
    'investment', 'investments', 'investment', 'investments', 
    'invested', 'investing', 'invest', 'invests', 'invested', 
    'investing', 'investor', 'investors', 'investing', 
    'invested', 'invest', 'invests', 'invested', 'investing', 
    'forward', 'looking', 'forward-looking', 'statements', 
    'statement', 'forward', 'looking', 'statements', 'statement']
    stopwords = set(STOPWORDS.union(custom_stopwords))
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)

    # Interpretation
    interpretation = "The word cloud above represents the most frequent words found in the report. Common terms such as 'company', 'business', 'report', 'financial', 'results', etc., have been excluded as they are not informative for understanding the key insights. This visualization provides a quick overview of the main topics or themes discussed in the report."
    
    return wordcloud, interpretation


def perform_sentiment_analysis(text):
    # Sentiment Analysis
    sentiment = TextBlob(text).sentiment.polarity
    sentiment_histogram = [TextBlob(sentence).sentiment.polarity for sentence in nltk.sent_tokenize(text)]

    # Interpretation
    if sentiment > 0:
        sentiment_interpretation = "The overall sentiment of the text is positive."
    elif sentiment == 0:
        sentiment_interpretation = "The overall sentiment of the text is neutral."
    else:
        sentiment_interpretation = "The overall sentiment of the text is negative."

    sentiment_distribution_interpretation = "The sentiment distribution histogram shows the distribution of sentiment polarity across sentences in the text. Positive sentiment values indicate positive opinions or emotions, while negative values indicate negative opinions or emotions. A value around 0 indicates neutral sentiment."

    return sentiment, sentiment_histogram, sentiment_interpretation, sentiment_distribution_interpretation

# Function to generate top words using Plotly
def generate_top_words_plotly(text):
    import nltk
    from nltk.corpus import stopwords

    # Download stopwords if not already downloaded
    nltk.download('stopwords')

    # Define custom stopwords
    custom_stopwords = ['$','2019','2020','2021','2022','2023','2023','2024','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','31.','.',',','31,','could','-','_']

    # Get the English stopwords
    stop_words = set(stopwords.words('english'))

    # Add custom stopwords to the set
    stop_words.update(custom_stopwords)

    # Remove stopwords from the text
    filtered_text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # Create a DataFrame for word frequency analysis
    word_freq = pd.Series(filtered_text.split()).value_counts()[:10].reset_index()
    word_freq.columns = ['Word', 'Frequency']

    # Create Plotly bar chart with custom colors
    fig = px.bar(word_freq, x='Word', y='Frequency', title='Top 10 Most Frequent Words',
                 color='Frequency', color_continuous_scale='Viridis')
    fig.update_xaxes(title_text='Word')
    fig.update_yaxes(title_text='Frequency')

    # Interpretation
    interpretation = "The bar chart displays the top 10 most frequent words extracted from the text, excluding common English stopwords as well as custom stopwords such as numbers, special characters, and common phrases. This visualization provides insight into the key terms that appear most frequently in the text, allowing for a clearer understanding of the prominent topics or themes discussed."
    
    return fig, interpretation

# Function to generate pie chart for sentiment analysis
def generate_pie_chart(text, file_name):
    sentiments = [TextBlob(sentence).sentiment.polarity for sentence in nltk.sent_tokenize(text)]
    # Categorize sentiment values into bins
    bins = [-0.5, 0, 0.5, 1]  # Define your own bins
    sentiment_categories = pd.cut(sentiments, bins=bins, labels=['Negative', 'Neutral', 'Positive'])

    # Count the number of reviews in each sentiment category
    sentiment_counts = sentiment_categories.value_counts()

    # Plot Pie Chart for Sentiment Distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'gray', 'green'])
    ax.set_title(f'Sentiment Distribution for {file_name}')
    plt.close(fig)  # Close the plot to avoid duplicate display in Streamlit

    # Interpretation
    interpretation = "The pie chart above represents the sentiment distribution of the text. Sentences are categorized into three sentiment categories: Negative, Neutral, and Positive, based on their sentiment polarity scores. This visualization provides an overview of the overall sentiment tone of the text."
    
    return fig, interpretation


def visualize_reports():
    # Process each PDF file in the User_Reports folder
    pdf_files = [file for file in os.listdir('User_Reports') if file.endswith('.pdf')]
    
    
    # Add 'Select a report to visualize' as the first option
    pdf_files.insert(0, 'Select a report to visualize')
    
    # Display dropdown menu for file selection
    selected_file = st.selectbox("Select a file:", pdf_files)

    
     # Display results if a file is selected
    if selected_file != 'Select a report to visualize':
        file_path = os.path.join('User_Reports', selected_file)
        text = extract_text_from_pdf(file_path)
        
        st.title(selected_file)

    
        # Generate Word Cloud
        wordcloud, wc_interpretation = generate_wordcloud(text)
        st.subheader('Word Cloud')
        st.image(wordcloud.to_array(), use_column_width=True)
        st.write(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{wc_interpretation}</p>', unsafe_allow_html=True)  # Display interpretation
        
        
        # Draw a horizontal line
        st.write('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
        
        # Perform Sentiment Analysis
        sentiment, sentiment_histogram, sentiment_interpretation, sentiment_distribution_interpretation = perform_sentiment_analysis(text)
        st.subheader('Sentiment Analysis')
        st.write(f'Overall Sentiment Polarity: {sentiment}')
        st.write(sentiment_interpretation)  # Display interpretation
        plt.figure(figsize=(7, 5))
        plt.hist(sentiment_histogram, bins=20, edgecolor='black', alpha=0.7)
        plt.title('Sentiment Distribution of Sentences')
        plt.xlabel('Sentiment')
        plt.ylabel('Frequency')
        st.pyplot(plt)  # Display sentiment histogram
        st.write(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{sentiment_distribution_interpretation}</p>', unsafe_allow_html=True)  # Display interpretation
        
        # Draw a horizontal line
        st.write('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
        
        # Generate Top Words using Plotly
        top_words_plotly, top_words_interpretation = generate_top_words_plotly(text)
        st.subheader('Top 10 Most Frequent Words')
        st.plotly_chart(top_words_plotly)
        st.write(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{top_words_interpretation}</p>', unsafe_allow_html=True)  # Display interpretation

        
        # Draw a horizontal line
        st.write('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
        
        
        # Generate Pie Chart for Sentiment Analysis
        pie_chart, pie_chart_interpretation = generate_pie_chart(text, selected_file)
        st.subheader('Pie Chart for Sentiment Distribution')
        st.pyplot(pie_chart)  # Display pie chart
        st.write(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{pie_chart_interpretation}</p>', unsafe_allow_html=True)  # Display interpretation
        
if __name__ == "__main__":
    visualize_reports()