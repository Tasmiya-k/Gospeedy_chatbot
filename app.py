# Home 🏠

import streamlit as st
from streamlit_card import card 
# from st_pages import Page, show_pages


# Import your page modules
from pages.Chatbot import main as chatbot_page
from pages.Visualizations import visualize_reports as reports_page
from pages.Faq import main as faq_page
from pages.News import News as news_page
from pages.ESG import main as esg_page
from pages.YT_Call import main as yt_page
from pages.Graphs import main as graphs_page
from pages.Risk_Profile import main as risk_profile


# # Define the navigation options
# PAGES = {
#     "Chatbot": chatbot_page,
#     "Visualize Reports": reports_page,
#     "Faq": faq_page,
#     "News": news_page
# }

# show_pages(
#     [
#         Page("app.py", "Home", "🏠"),
#         Page("pages/Chatbot.py","Chatbot","📃"),
#         Page("pages/Visualizations.py","Visualize the Reports","📊"),
#         Page("pages/Faq.py","Frequently Asked Questions","🤔"),
#         Page("pages/News.py","Latest News and Stock Data","📰"),
#     ]
# )

# Create a function to render the selected page
def Home():
    # Render the Chatbot page by default
    st.title("Welcome to the Annual Public Reports Chatbot 🤖")
    st.write("Explore what you can do!")
    # Define a list of dictionaries containing card information
    cards = [
        {
            "title": "Search annual public reports from all over the web",
            "text":"",
            "image": "https://images.unsplash.com/photo-1586769852836-bc069f19e1b6?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Real-time FAQs generation",
            "text":"",
            "image": "https://images.unsplash.com/photo-1477281765962-ef34e8bb0967?q=80&w=1933&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Chat with multiple annual public reports",
            "text":"",
            "image": "https://images.unsplash.com/photo-1488509082528-cefbba5ad692?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Get accurate answers to all your questions",
            "text":"Straight-forward, Open-ended, Suggestive, Analytical, Predictive",
            "image": "https://images.unsplash.com/photo-1518644730709-0835105d9daa?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Response-specific data visualization",
            "text":"",
            "image": "https://images.unsplash.com/photo-1635236190542-d43e4d4b9e4b?q=80&w=2069&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Report-specific data visualization",
            "text":"",
            "image": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Get the latest news about your desired company",
            "text":"",
            "image": "https://images.unsplash.com/photo-1546422904-90eab23c3d7e?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        },
        {
            "title": "Get the real-time stock data",
            "text":"",
            "image": "https://images.unsplash.com/photo-1579407364450-481fe19dbfaa?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "url": ""
        }
        # Add more dictionaries for additional cards
    ]

    # Create two columns for the cards
    col1, col2 = st.columns(2)

    # Loop through the list of dictionaries to generate cards dynamically
    for i, card_info in enumerate(cards):
        if i % 2 == 0:
            column = col1
        else:
            column = col2

        with column:
            card(title=card_info["title"], text=card_info["text"], image=card_info["image"], url=card_info["url"])



# Run the app
if __name__ == "__main__":
    Home()
