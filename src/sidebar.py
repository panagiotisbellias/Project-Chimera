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

    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key

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
    st.sidebar.markdown(
        """
        <style>
        .sidebar-button {
            display: flex;
            align-items: center;
            justify-content: center;
            background: #ffffff;
            color: #2C3E50 !important;
            padding: 6px 12px;
            margin: 6px 0;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.9rem;
            text-decoration: none !important;
            border: 1px solid #E0E0E0;
            transition: all 0.2s ease-in-out;
        }
        .sidebar-button:hover {
            background: #F8F9FA;
            border-color: #D0D0D0;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .sidebar-button img {
            height: 16px;
            margin-right: 6px;
        }
        </style>

        <a href="https://github.com/akarlaraytu/Project-Chimera" target="_blank" class="sidebar-button">⭐ Star on GitHub</a>
        <a href="https://discord.gg/t6M4yyeRQc" target="_blank" class="sidebar-button">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/discord.svg" alt="Discord"/>
            Join Discord
        </a>
        <a href="https://medium.com/@akarlaraytu" target="_blank" class="sidebar-button">✍️ Support on Medium</a>
        """,
        unsafe_allow_html=True
    )


    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='text-align:center; font-size:0.8rem; color:#95A5A6;'>© 2025 Project Chimera</p>", unsafe_allow_html=True)
