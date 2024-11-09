from typing import List, Dict, Any
import requests
import os
import re
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rag_search import search_google, highlight_citations, ReportGenerator

import streamlit as st

def main():
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
    gg_api_key = st.secrets["GG_API_KEY"]
    gg_cse_id = st.secrets["CSE_ID"]

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024
    )

    # Webpage Logo
    col1, col2, col3 = st.columns([0.35, 0.3, 0.35])
    with col2:
        st.markdown(
            """
            <style>
            .center-image {
                display: flex;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.container():
            st.markdown('<div class="center-image">', unsafe_allow_html=True)
            st.image("banner.png", width=180)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Web Title
    st.markdown("<h1 style='text-align: center; margin-top: 0;'>EdSearch</h1>", unsafe_allow_html=True)

    # Initialize session state for topic and output_len if they don't exist
    if "topic" not in st.session_state:
        st.session_state["topic"] = ""
    if "output_len" not in st.session_state:
        st.session_state["output_len"] = "1000"
    
    # Input columns
    col1, col2 = st.columns([3, 1])

    # Topic box on the left column
    with col1:
        st.session_state["topic"] = st.text_input("Enter Topic:", st.session_state["topic"])
    
    # Length box on the right column
    with col2:
        st.session_state["output_len"] = st.text_input("~Length:", st.session_state["output_len"])
    
    # Search and Reset buttons
    col_search, col_reset = st.columns([0.3, 0.3])
    
    with col_search:
        search_clicked = st.button("Search")

    with col_reset:
        reset_clicked = st.button("Reset")
    
    # Handle Search button click
    if search_clicked:
        if st.session_state["topic"] and st.session_state["output_len"].isdigit():
            # Generate the article
            # Initialize report generator
            report_gen = ReportGenerator(embeddings, st.session_state["output_len"])

            # Generate report
            result = report_gen.generate_report(st.session_state["topic"])

            st.session_state.article = result["report"]
            
            # Show the generated article
            with st.expander("Click to View full Article"):
                st.markdown(result["report"], unsafe_allow_html=True)
        else:
            st.warning("Please enter a valid topic and word count.")

    # Handle Reset button click
    if reset_clicked:
        # del st.session_state["topic"]
        # del st.session_state["output_len"]
        if 'article' in st.session_state:
            del st.session_state["article"]
        st.session_state["topic"] = ""
        st.session_state["output_len"] = "1000"

if __name__ == "__main__":
    main()
