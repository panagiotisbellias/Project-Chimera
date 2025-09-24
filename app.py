# app.py
import sys
import os
import streamlit as st

from src.ui import hide_default_sidebar_nav
from src.sidebar import render_sidebar

render_sidebar()
hide_default_sidebar_nav()

st.set_page_config(
    page_title="Project Chimera - Main Menu",
    page_icon="🏛️",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)

hide_default_sidebar_nav()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Main_Menu import main

if __name__ == "__main__":
    main()