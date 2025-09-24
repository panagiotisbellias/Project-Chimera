# src/sidebar.py
import os
import streamlit as st
import requests

def validate_openai_key(key: str) -> bool:
    """
    OpenAI API Key Validation
    """
    try:
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False

def render_sidebar():
    
    st.sidebar.markdown("<h2 style='text-align:center; color:#2C3E50;'>🏛️ Project Chimera</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align:center; font-size:0.9rem; color:#7F8C8D;'>Multi-Agent Simulation Suite</p>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Manual page links
    st.sidebar.page_link("app.py", label="🏠 Main Menu")
    st.sidebar.page_link("pages/1_🔬_Adaptive_Strategy_Lab.py", label="🔬 Strategy Lab")
    st.sidebar.page_link("pages/2_⚔️_Colosseum.py", label="⚔️ Colosseum")

    st.sidebar.markdown("---")

    api_key = st.sidebar.text_input(
        "Enter your API key",
        type="password",
        value=st.session_state.get("OPENAI_API_KEY", ""),
        placeholder="sk-...",
        key="openai_api_key_global"
        )

    # 🔹 Anında yaz
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key

    # Buton sadece sayfayı yenilemek için
    if st.sidebar.button("Apply API Key"):
        st.rerun()

    if api_key:
        if validate_openai_key(api_key):
            st.sidebar.success("API key is valid ✅")
        else:
            st.sidebar.error("Invalid API key ❌")
    else:
        st.sidebar.warning("Please enter your API key.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='text-align:center; font-size:0.8rem; color:#95A5A6;'>© 2025 Project Chimera</p>", unsafe_allow_html=True)
