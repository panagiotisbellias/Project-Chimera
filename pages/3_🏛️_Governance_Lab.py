# pages/3_🏛️_Governance_Lab.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
import os
import sys
import time

# Projenin ana dizinini path'e ekleyerek src klasöründeki modüllere erişim sağlıyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import CausalEngineV6_UseReady
from src.sidebar import render_sidebar
from src.ui import hide_default_sidebar_nav

# --- SAYFA YAPILANDIRMASI VE KENAR ÇUBUĞU ---
st.set_page_config(
    page_title="Governance Lab",
    page_icon="🏛️",
    layout="wide"
)
render_sidebar()
hide_default_sidebar_nav()

# --- YENİ AYDINLIK TEMA İÇİN CSS CİLASI ---
st.markdown("""
<style>
    :root {
        --primary-color: #E6007A; /* Polkadot Pink */
        --background-color: #FFFFFF; /* Hafif gri, modern bir arkaplan */
        --secondary-background-color: #FFFFFF; /* Kartlar için beyaz */
        --text-color: #1E1E1E; /* Saf siyah yerine koyu gri */
        --secondary-text-color: #555555;
    }
    .stApp {
        background-color: var(--background-color);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }
    /* Kartların (container) tasarımını güzelleştirme */
    [data-testid="stVerticalBlock"] .st-emotion-cache-1jicfl2 {
        background-color: var(--secondary-background-color);
        border: 1px solid #EAEAEA;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        height: 100%; /* Kutuların boyunu eşitleme */
    }
    /* Ana (primary) butonu Polkadot Pembesi yapma */
    .stButton button[data-testid="baseButton-primary"] {
        background-color: var(--primary-color);
        color: white;
        border: none;
    }
    .stButton button[data-testid="baseButton-primary"]:hover {
        background-color: #B80062;
        color: white;
        border: none;
    }
    /* İlerleme çubuğunu Polkadot Pembesi yapma */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# OpenAI API anahtarının girilip girilmediğini kontrol et
if "OPENAI_API_KEY" not in st.session_state or not st.session_state["OPENAI_API_KEY"]:
    st.error("Lütfen başlamadan önce kenar çubuğundan OpenAI API anahtarınızı girin.")
    st.stop()

# --- OTURUM DURUMU (SESSION STATE) YÖNETİMİ ---
if "causal_engine" not in st.session_state:
    try:
        st.session_state.causal_engine = CausalEngineV6_UseReady()
    except FileNotFoundError as e:
        st.error(f"Kritik Hata: Causal Engine modeli bulunamadı. Hata: {e}")
        st.stop()

if "gov_proposals" not in st.session_state:
    proposals_data = {
        "🚀 Proposal A: Aggressive Growth": {
            "description": "Cut prices significantly and boost ad spend to capture market share quickly.",
            "action": {"price_change": -0.20, "ad_spend": 3000.0},
            "aye_votes": 0, "nay_votes": 0
        },
        "⚖️ Proposal B: Balanced Approach": {
            "description": "Maintain current price levels while moderately increasing ad spend for steady growth.",
            "action": {"price_change": 0.0, "ad_spend": 1500.0},
            "aye_votes": 0, "nay_votes": 0
        },
        "💰 Proposal C: Profitability Focus": {
            "description": "Slightly increase prices and reduce ad spend to maximize short-term profit margins.",
            "action": {"price_change": +0.10, "ad_spend": 500.0},
            "aye_votes": 0, "nay_votes": 0
        }
    }
    st.session_state.gov_proposals = proposals_data

if "gov_simulation_result" not in st.session_state: st.session_state.gov_simulation_result = {}
if "community_simulated" not in st.session_state: st.session_state.community_simulated = False


# --- ARAYÜZ (UI) TASARIMI ---
st.title("🏛️ Governance Lab")
st.markdown("### Simulating On-Chain Decisions with Causal AI")
with st.expander("ℹ️ About This Lab", expanded=False):
    st.markdown("This prototype demonstrates how Chimera brings data-driven foresight to the **'blind voting'** problem in decentralized governance. Simulate proposals, use your token power to vote, and see how a community might react.")

st.divider()

# --- KULLANICI KONTROL PANELİ ---
st.subheader("Your Governance Control Panel")
user_tokens = st.slider("Your Voting Power (Tokens)", 100, 100000, 25000, 100)

# --- OYLAMA SONUÇLARI PANELİ ---
st.subheader("Live Voting Results")
total_votes = sum(p['aye_votes'] + p['nay_votes'] for p in st.session_state.gov_proposals.values())
if total_votes > 0:
    for title, data in st.session_state.gov_proposals.items():
        aye_percent = (data['aye_votes'] / total_votes) * 100 if total_votes > 0 else 0
        st.markdown(f"**{title}** - Total Aye Power: **{int(data['aye_votes']):,}**")
        st.progress(int(aye_percent))
else:
    st.info("No votes have been cast yet. Simulate a proposal and cast your vote!")

st.divider()

# --- ÖNERİLER (PROPOSALS) BÖLÜMÜ ---
CONTEXT = {"price": 100.0, "brand_trust": 0.7, "weekly_ad_spend": 1000.0, "season_phase": 10}
st.subheader("Treasury Vote: Choose an Economic Strategy")

col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

for i, (title, data) in enumerate(st.session_state.gov_proposals.items()):
    with columns[i]:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.caption(f"{data['description']}")
            st.code(f"Price Change: {data['action']['price_change']*100:+.0f}%\nAd Spend: ${data['action']['ad_spend']:,.0f}", language="text")
            
            if st.button("Simulate Impact", key=f"sim_{i}", use_container_width=True):
                with st.spinner("Chimera is analyzing..."):
                    engine = st.session_state.causal_engine
                    estimates = engine.estimate_causal_effect(action=data['action'], context=CONTEXT)
                    explanation = engine.explain_decision(context=CONTEXT, action=data['action'])
                    st.session_state.gov_simulation_result = {"title": title, "estimates": estimates, "explanation": explanation}
            
            with st.expander("Cast Your Vote"):
                conviction_map = {"1x (No Lock)": 1, "2x (Lock for 8 weeks)": 2, "4x (Lock for 32 weeks)": 4, "6x (Lock for 128 weeks)": 6}
                conviction = st.selectbox("Vote Conviction", options=conviction_map.keys(), key=f"conv_{i}", label_visibility="collapsed")
                
                if st.button("Vote AYE", key=f"vote_{i}", use_container_width=True, type="primary"):
                    vote_power = user_tokens * conviction_map[conviction]
                    st.session_state.gov_proposals[title]['aye_votes'] += vote_power
                    st.toast(f"You voted for {title} with {vote_power:,} power!", icon="🗳️")
                    time.sleep(0.5) 
                    st.rerun()

st.divider()

# --- TOPLULUK SİMÜLASYONU VE CHIMERA ANALİZ BÖLÜMÜ ---
analysis_col1, analysis_col2 = st.columns([1, 2])

with analysis_col1:
    st.subheader("Community Simulation")
    if not st.session_state.community_simulated:
        if st.button("Simulate 100 Community Voters", use_container_width=True):
            if not st.session_state.gov_simulation_result:
                st.warning("Please simulate at least one proposal before simulating the community.")
            else:
                with st.spinner("Simulating community reaction..."):
                    sim_results = {}
                    for p_title, p_data in st.session_state.gov_proposals.items():
                        sim_results[p_title] = st.session_state.causal_engine.estimate_causal_effect(action=p_data['action'], context=CONTEXT)
                    
                    most_profitable = max(sim_results, key=lambda k: sim_results[k]['estimated_long_term_value'])
                    most_trustworthy = max(sim_results, key=lambda k: sim_results[k]['simulated_trust_change'])

                    for _ in range(50): 
                        st.session_state.gov_proposals[most_profitable]['aye_votes'] += np.random.randint(100, 5000)
                    for _ in range(30):
                        st.session_state.gov_proposals[most_trustworthy]['aye_votes'] += np.random.randint(100, 5000)
                    for _ in range(20):
                        random_choice = np.random.choice(list(st.session_state.gov_proposals.keys()))
                        st.session_state.gov_proposals[random_choice]['aye_votes'] += np.random.randint(100, 5000)
                
                st.session_state.community_simulated = True
                st.toast("Community simulation complete!", icon="👥")
                time.sleep(0.5)
                st.rerun()
    else:
        st.success("Community simulation has been run.")

with analysis_col2:
    st.subheader("Chimera's Causal Analysis")
    if st.session_state.gov_simulation_result.get("title"):
        result = st.session_state.gov_simulation_result
        st.markdown(f"##### Analysis for: *{result['title']}*")
        
        estimates = result['estimates']
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Predicted Long-Term Value", f"${estimates.get('estimated_long_term_value', 0):,.0f}")
        metric_col2.metric("Predicted Profit Change", f"${estimates.get('predicted_short_term_profit', 0):,.0f}")
        metric_col3.metric("Brand Trust Change", f"{estimates.get('simulated_trust_change', 0):.3f}")

        shap_obj = result["explanation"]["shap_object"]
        # SHAP GRAFİĞİ İÇİN AYDINLIK TEMA AYARLARI
        fig, ax = plt.subplots(figsize=(8, 3.5)); 
        shap.plots.waterfall(shap_obj[0], show=False, max_display=6)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Click a 'Simulate Impact' button to see Chimera's detailed analysis here.")