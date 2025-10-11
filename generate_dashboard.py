# =============================================================================
# generate_dashboard.py (v2.7 - Premium Layout & Disclaimer)
#
# Description:
#   This final version refines the layout for a more premium feel. The
#   disclaimer is moved to a prominent position at the top to avoid any
#   layout conflicts, and the 'Last Actions' section now has dedicated space.
# =============================================================================

print("--- RUNNING FINAL ATOMIC WRITE VERSION ---")

import matplotlib
matplotlib.use('Agg')

import sys

import os
import time
import datetime as dt
from io import BytesIO

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import json
from alpaca_trade_api.rest import APIError

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL_ALPACA = "BTC/USD" # For data fetching
SYMBOL_DISPLAY = "BTCUSD" # For display and position fetching
OUTPUT_IMAGE_PATH = "live_dashboard.png"
UPDATE_INTERVAL_SECONDS = 30 # Refresh every 30 seconds

# --- SETUP ---
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- STYLING ---
BG_COLOR = '#0d1117'
TEXT_COLOR = '#E0E0E0'
SUBTEXT_COLOR = '#A0A0A0'
GOLD_COLOR = '#ca8a04'
GREEN_COLOR = '#22c55e'
RED_COLOR = '#ef4444'


try:
    FONT_BOLD = ImageFont.truetype("Poppins-Bold.ttf", 32)
    FONT_MEDIUM = ImageFont.truetype("Poppins-Medium.ttf", 24)
    FONT_REGULAR = ImageFont.truetype("Poppins-Regular.ttf", 18)
    FONT_SMALL = ImageFont.truetype("Poppins-Regular.ttf", 14)
    FONT_TINY = ImageFont.truetype("Poppins-Regular.ttf", 12)
except IOError:
    print("Warning: Poppins font not found. Using default fonts.")
    FONT_BOLD = FONT_MEDIUM = FONT_REGULAR = FONT_SMALL = FONT_TINY = ImageFont.load_default()

def format_currency(value):
    return f"${value:,.2f}"

def create_btc_chart_image(data_df):
    """ (NO TRANSPARENCY FINAL VERSION) Creates a stylized BTC price chart. """
    if data_df.empty:
        return None

    # Aggregate 1-minute data to 5-minute data for a cleaner chart
    if len(data_df) > 10:
        data_df = data_df.resample('5Min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    mc = mpf.make_marketcolors(up=GREEN_COLOR, down=RED_COLOR,
                               wick={'up':GREEN_COLOR, 'down':RED_COLOR},
                               edge={'up':GREEN_COLOR, 'down':RED_COLOR},
                               volume='inherit')
    
    # Set the background color directly in the style
    s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds',
                           gridstyle='-', gridcolor='#2c2c2c', facecolor=BG_COLOR)

    fig, axlist = mpf.plot(data_df, type='candle', style=s,
                           volume=False,
                           figsize=(10.5, 4.5),
                           returnfig=True)
    
    ax = axlist[0]
    # Explicitly set background colors to be safe
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    ax.set_ylabel("Price ($)", color=SUBTEXT_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), color=SUBTEXT_COLOR)
    plt.setp(ax.get_yticklabels(), color=SUBTEXT_COLOR)

    buf = BytesIO()
    # Save with transparency turned OFF
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=BG_COLOR)
    buf.seek(0)
    chart_img = Image.open(buf)
    plt.close(fig)
    return chart_img

def create_dashboard_image(portfolio_data, chart_img):
    """ (NO TRANSPARENCY FINAL VERSION) Creates the final dashboard image. """
    img = Image.new('RGB', (1280, 720), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # --- HEADER ---
    draw.text((50, 40), "Project Chimera", font=FONT_BOLD, fill=TEXT_COLOR)
    draw.text((50, 85), "Live Quant Trading", font=FONT_MEDIUM, fill=GOLD_COLOR)
    draw.ellipse((1140, 52, 1160, 72), fill=RED_COLOR, outline=RED_COLOR)
    draw.text((1170, 45), "LIVE", font=FONT_MEDIUM, fill=TEXT_COLOR)

    # --- DISCLAIMER ---
    disclaimer_text = "This is an experimental research project. All trades are simulated (paper trading). Not financial advice."
    draw.text((1230, 95), disclaimer_text, font=FONT_SMALL, fill="#888888", anchor="ra", align="right")

    # --- LEFT PANEL: METRICS ---
    pnl_color = GREEN_COLOR if portfolio_data.get('pnl', 0) >= 0 else RED_COLOR
    y = 180
    draw.text((50, y), "Portfolio Value", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    draw.text((50, y + 30), format_currency(portfolio_data.get('value', 0)), font=FONT_BOLD, fill=TEXT_COLOR)
    y += 100
    draw.text((50, y), "Unrealized P&L (Session)", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    draw.text((50, y + 30), f"{portfolio_data.get('pnl', 0):+.2f} ({portfolio_data.get('pnl_pct', 0):+.2f}%)", font=FONT_BOLD, fill=pnl_color)
    y += 100
    draw.text((50, y), "Current Position", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    draw.text((50, y + 30), f"{portfolio_data.get('qty', 0):.4f} BTC", font=FONT_BOLD, fill=TEXT_COLOR)
    y += 100
    draw.text((50, y), "Sharpe / Max Drawdown", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    draw.text((50, y + 30), "Calculating...", font=FONT_BOLD, fill=TEXT_COLOR)

    # --- RIGHT PANEL: BTC CHART ---
    if chart_img:
        # Simple paste, no mask, no transparency issues.
        img.paste(chart_img, (400, 150))
    
    draw.text((430, 120), "BTC/USD Price Chart (Last 60 Days)", font=FONT_MEDIUM, fill=TEXT_COLOR)

    # --- BOTTOM RIGHT: AGENT'S LAST CAUSAL ANALYSIS ---
    draw.line((430, 560, 1230, 560), fill="#4A4A4A", width=1)
    draw.text((430, 570), "Agent's Last Causal Analysis:", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    agent_analysis = portfolio_data.get('agent_analysis')
    if agent_analysis and 'scenarios' in agent_analysis:
        y_scenario = 605
        for scenario in agent_analysis.get('scenarios', [])[:4]:
            hypo_text = scenario.get('hypothesis', 'N/A')
            validation = scenario.get('validation', 'Pending')
            impact = scenario.get('impact')
            status_char = "✅" if validation == "Valid" else "❌"
            hypo_color = TEXT_COLOR if validation == "Valid" else "#888888"
            draw.text((450, y_scenario), f"{hypo_text.ljust(15)} {status_char}", font=FONT_SMALL, fill=hypo_color)
            if impact is not None and validation == "Valid":
                impact_text = f"Est. Impact: {impact:+.2%}"
                impact_color = GREEN_COLOR if impact > 0 else RED_COLOR if impact < 0 else GOLD_COLOR
                draw.text((750, y_scenario), impact_text, font=FONT_SMALL, fill=impact_color)
            y_scenario += 20
    else:
        draw.text((450, 600), "Awaiting first agent analysis...", font=FONT_SMALL, fill=SUBTEXT_COLOR)

    # --- BOTTOM LEFT: LAST EXECUTED TRADES ---
    draw.line((50, 560, 380, 560), fill="#4A4A4A", width=1)
    draw.text((50, 570), "Last Executed Trades:", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    actions_list = portfolio_data.get('last_actions', ["Awaiting first action..."])
    y_action = 605
    for action_text in actions_list[:3]:
        draw.text((50, y_action), action_text, font=FONT_REGULAR, fill=TEXT_COLOR)
        y_action += 25 
    
    # --- FOOTER ---
    draw.text((1230, 700), f"Last Update: {portfolio_data.get('timestamp', 'N/A')} (UTC)", font=FONT_TINY, fill="#4A4A4A", anchor="rs")

    # --- ATOMIC FILE WRITE ---
    temp_output_path = OUTPUT_IMAGE_PATH + ".tmp"
    img.save(temp_output_path, "PNG")
    os.replace(temp_output_path, OUTPUT_IMAGE_PATH)
    
    print(f"Dashboard image updated at {portfolio_data.get('timestamp')}")

def fetch_data():
    """
    (LIVE CHART VERSION)
    Fetches all necessary data. For the chart, it now fetches 1-minute bars
    for the current day to create a more dynamic, "live" chart.
    """
    try:
        # --- ROBUST API CALLS ---
        account = api.get_account()

        qty, pnl, pnl_pct = 0.0, 0.0, 0.0
        try:
            position = api.get_position(SYMBOL_DISPLAY)
            qty = float(position.qty)
            pnl = float(position.unrealized_pl)
            pnl_pct = float(position.unrealized_plpc) * 100
        except APIError as e:
            if "position not found" not in str(e).lower():
                print(f"API Error fetching position (will default to 0): {e}")

        # --- DYNAMIC CHART DATA FETCH ---
        # Fetch 1-minute bars for the current day (UTC timezone)
        today_utc = dt.datetime.now(dt.timezone.utc)
        start_date = today_utc.strftime('%Y-%m-%d') # Bugünün başlangıcı

        market_data_df = api.get_crypto_bars(
            SYMBOL_ALPACA,
            '1Min', # Zaman aralığını 1 dakikaya düşürdük
            start=start_date
        ).df
        market_data_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        # --- END DYNAMIC CHART DATA FETCH ---

        # --- ROBUST FILE PATH LOGIC ---
        agent_analysis = None
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "last_cycle_report.json")
            with open(json_path, "r") as f:
                agent_analysis = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # --- ROBUST ORDER FETCHING ---
        last_actions = []
        try:
            orders = api.list_orders(status='closed', limit=3, direction='desc')
            if orders:
                for order in orders:
                    if order.filled_at:
                        filled_time = pd.to_datetime(order.filled_at).tz_convert('UTC')
                        action_str = f"• {order.side.upper()} {float(order.filled_qty):.4f} @ {format_currency(float(order.filled_avg_price))} ({filled_time.strftime('%b %d, %H:%M')})"
                        last_actions.append(action_str)
        except Exception as e:
            print(f"Could not fetch last orders: {e}")

        if not last_actions:
            last_actions.append("Awaiting first agent action...")

        return {
            "value": float(account.equity),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "qty": qty,
            "last_actions": last_actions,
            "timestamp": dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M:%S'),
            "market_data": market_data_df,
            "agent_analysis": agent_analysis
        }

    except Exception as e:
        print(f"A CRITICAL ERROR occurred during the main fetch_data cycle: {e}")
        return {
            "value": 0, "pnl": 0, "pnl_pct": 0, "qty": 0,
            "last_actions": ["API Connection Error"],
            "timestamp": dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M:%S'),
            "market_data": pd.DataFrame(),
            "agent_analysis": None
        }


if __name__ == "__main__":
    while True:
        try:
            print("--- Starting Dashboard Generation Cycle ---") 
            live_data = fetch_data()
            create_dashboard_image(live_data['market_data'], live_data)
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
        
        print(f"--- Cycle Complete. Sleeping for {UPDATE_INTERVAL_SECONDS} seconds... ---")
        time.sleep(UPDATE_INTERVAL_SECONDS)

