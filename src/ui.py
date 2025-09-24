# src/ui.py
import streamlit as st

def hide_default_sidebar_nav():
    st.markdown(
        """
        <style>
        /* Hide Streamlit's automatic sidebar page list */
        [data-testid="stSidebarNav"] { display: none !important; }
        /* Optional: tighten sidebar spacing */
        [data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True
    )
