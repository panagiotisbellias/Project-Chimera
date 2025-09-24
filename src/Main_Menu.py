import streamlit as st
import os

if "OPENAI_API_KEY" not in st.session_state or not st.session_state["OPENAI_API_KEY"]:
    st.error("Please enter your OpenAI API key in the sidebar before starting.")
    st.stop()


def main():
    st.set_page_config(
        page_title="Project Chimera - Main Menu",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Welcome to Project Chimera, an ecosystem for strategic AI agents.",
            "Get Help": None,
            "Report a bug": None,
        }
    )
    

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_path = os.path.join(PROJECT_ROOT, "assets", "main_menu_banner.png")
    if os.path.exists(image_path):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.image(image_path, use_container_width=True)

    st.markdown("<h1 style='text-align: center;'>Welcome to Project Chimera</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-weight: normal;'>An Ecosystem for Trustworthy & Strategic AI Agents</h3>", unsafe_allow_html=True)
    st.info(
        "**Welcome!** This is the central hub for Project Chimera's interactive applications. "
        "Please select an experience below or use the navigation panel on the left."
    )
    st.divider()

    col1, col2 = st.columns(2)

    #Adaptive Strategy Lab
    with col1:
        # st.container(border=True)
        with st.container(border=True):
            st.subheader("🔬 Adaptive Strategy Lab")
            st.markdown(
                "A deep-dive analysis tool for a single agent. "
                "Define a strategic goal, watch the agent think, and explore its decisions with an **interactive XAI dashboard**."
            )
            st.caption("*(Use this for detailed, single-agent analysis.)*")
            
            st.page_link(
                "pages/1_🔬_Adaptive_Strategy_Lab.py", 
                label="Launch Strategy Lab",
                icon="🔬",
                use_container_width=True
            )

    #The Colosseum
    with col2:
        with st.container(border=True):
            st.subheader("⚔️ The Colosseum")
            st.markdown(
                "An interactive battle arena! Assemble a team of AI gladiators with custom names and doctrines, "
                "and watch them compete against each other in a **live, gamified simulation**."
            )
            st.caption("*(Use this for multi-agent competitive analysis and fun!)*")
            
            st.page_link(
                "pages/2_⚔️_Colosseum.py",
                label="Enter The Colosseum",
                icon="⚔️",
                use_container_width=True
            )
if __name__ == "__main__":
    main()