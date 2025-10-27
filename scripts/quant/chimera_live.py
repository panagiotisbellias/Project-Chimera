# chimera_dashboard_animated.py - DASHBOARD


import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import json
import pickle
import argparse
import subprocess
import math
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Dict, List, Any, Optional
from collections import deque
import threading

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import APIError

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL_ALPACA = "BTC/USD"
SYMBOL_DISPLAY = "BTCUSD"

# Use absolute path for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "live_dashboard.png")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames")

FRAME_BUFFER_SIZE = 15
UPDATE_INTERVAL_SECONDS = 1
DATA_FETCH_INTERVAL = 30  
NEXT_RUN_TIME = "00:00"  # UTC
STARTING_CAPITAL = 100000.0

# File paths (all absolute)
PERFORMANCE_HISTORY_PATH = os.path.join(SCRIPT_DIR, "performance_history.pkl")
TRADE_HISTORY_PATH = os.path.join(SCRIPT_DIR, "trade_history.json")
ACTIVITY_LOG_PATH = os.path.join(SCRIPT_DIR, "trader_activity.log")

# Animation state
animation_frame = 0

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

g_cached_data = None
g_data_lock = threading.Lock()
g_last_frame_heartbeat = time.time()

# =============================================================================
# STYLING
# =============================================================================

BG_COLOR = '#0a0e14'
PANEL_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
SUBTEXT_COLOR = '#8b949e'
GOLD_COLOR = '#fbbf24'
GREEN_COLOR = '#3fb950'
RED_COLOR = '#f85149'
BLUE_COLOR = '#58a6ff'
PURPLE_COLOR = '#bc8cff'

# Fonts
try:
    FONT_TITLE = ImageFont.truetype("Poppins-Bold.ttf", 36)
    FONT_HEADER = ImageFont.truetype("Poppins-Bold.ttf", 24)
    FONT_MEDIUM = ImageFont.truetype("Poppins-Medium.ttf", 20)
    FONT_REGULAR = ImageFont.truetype("Poppins-Regular.ttf", 16)
    FONT_SMALL = ImageFont.truetype("Poppins-Regular.ttf", 13)
    FONT_TINY = ImageFont.truetype("Poppins-Regular.ttf", 11)
    FONT_MONO = ImageFont.truetype("Poppins-Regular.ttf", 32)  # For timer
except IOError:
    print("⚠️  Poppins fonts not found. Using defaults.")
    FONT_TITLE = FONT_HEADER = FONT_MEDIUM = FONT_REGULAR = FONT_SMALL = FONT_TINY = FONT_MONO = ImageFont.load_default()

# =============================================================================
# ANIMATION HELPERS
# =============================================================================

def get_pulse_alpha(frame: int, speed: float = 0.05) -> float:
    """Get pulsing alpha value (0.5 to 1.0)"""
    return 0.5 + 0.5 * abs(math.sin(frame * speed))

def get_gradient_color(frame: int, base_color: str) -> str:
    """Get animated gradient color"""
    # Parse hex color
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)
    
    # Add subtle animation
    offset = int(10 * math.sin(frame * 0.02))
    r = max(0, min(255, r + offset))
    g = max(0, min(255, g + offset))
    b = max(0, min(255, b + offset))
    
    return f"#{r:02x}{g:02x}{b:02x}"

def get_next_run_countdown() -> Dict[str, int]:
    """Calculate time until next bot run (returns dict with h,m,s)"""
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if now.hour >= 0 and now.minute >= 0:
        next_run += timedelta(days=1)
    
    delta = next_run - now
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    seconds = delta.seconds % 60
    
    return {'hours': hours, 'minutes': minutes, 'seconds': seconds, 'total_seconds': delta.seconds}

# =============================================================================
# DATA LOADING (Same as before)
# =============================================================================

def format_currency(value: float) -> str:
    return f"${value:,.2f}"

def load_trade_history() -> List[Dict]:
    if os.path.exists(TRADE_HISTORY_PATH):
        try:
            with open(TRADE_HISTORY_PATH, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_trade_history(history: List[Dict]):
    try:
        with open(TRADE_HISTORY_PATH, 'w') as f:
            json.dump(history[-50:], f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save trade history: {e}")

def update_trade_history(orders: List) -> List[Dict]:
    history = load_trade_history()
    existing_ids = {t['order_id'] for t in history}
    
    for order in orders:
        if order.id not in existing_ids and order.filled_at:
            history.append({
                'order_id': order.id,
                'timestamp': order.filled_at.isoformat(),
                'side': order.side,
                'qty': float(order.filled_qty),
                'price': float(order.filled_avg_price),
                'value': float(order.notional) if order.notional else 0.0
            })
    
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    save_trade_history(history)
    return history[:10]

def load_performance_history() -> Dict[str, List]:
    if os.path.exists(PERFORMANCE_HISTORY_PATH):
        try:
            with open(PERFORMANCE_HISTORY_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {'timestamps': [], 'values': [], 'daily_returns': []}

def save_performance_history(history: Dict):
    try:
        with open(PERFORMANCE_HISTORY_PATH, 'wb') as f:
            pickle.dump(history, f)
    except Exception as e:
        print(f"⚠️  Could not save performance: {e}")

def update_performance_history(current_value: float) -> Dict:
    history = load_performance_history()
    now = datetime.now(timezone.utc)
    
    history['timestamps'].append(now)
    history['values'].append(current_value)
    
    if len(history['timestamps']) > 12960:
        history['timestamps'] = history['timestamps'][-12960:]
        history['values'] = history['values'][-12960:]
    
    if len(history['values']) > 1:
        returns = np.diff(history['values']) / np.array(history['values'][:-1])
        history['daily_returns'] = returns.tolist()
    
    save_performance_history(history)
    return history

def calculate_sharpe_ratio(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    if np.std(returns_array) == 0:
        return 0.0
    return float(np.sqrt(365) * np.mean(returns_array) / np.std(returns_array))

def calculate_max_drawdown(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    values_array = np.array(values)
    cummax = np.maximum.accumulate(values_array)
    drawdown = (values_array - cummax) / cummax
    return float(np.min(drawdown) * 100)

def load_agent_analysis() -> Optional[Dict]:
    try:
        with open("last_cycle_report.json", "r") as f:
            return json.load(f)
    except:
        return None

def read_recent_logs(log_file: str = "trader_activity.log", lines: int = 5) -> List[str]:
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:]]
    except:
        return ["System ready - No activity yet"]

# =============================================================================
# CHART GENERATION
# =============================================================================

def create_btc_chart_image(data_df: pd.DataFrame) -> Optional[Image.Image]:
    if data_df.empty:
        return None

    mc = mpf.make_marketcolors(
        up=GREEN_COLOR, down=RED_COLOR,
        wick={'up': GREEN_COLOR, 'down': RED_COLOR},
        edge={'up': GREEN_COLOR, 'down': RED_COLOR},
        volume='inherit'
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style='nightclouds',
        gridstyle='-',
        gridcolor='#1c2128',
        facecolor=PANEL_COLOR
    )

    fig, axlist = mpf.plot(
        data_df,
        type='candle',
        style=s,
        volume=False,
        figsize=(7, 3.5),
        returnfig=True,
        warn_too_much_data=10000,
        datetime_format='%H:%M'  
    )
    
    ax = axlist[0]
    fig.patch.set_facecolor(PANEL_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.set_ylabel("Price ($)", color=SUBTEXT_COLOR, fontsize=10)
    
    
    plt.setp(ax.get_xticklabels(), color=SUBTEXT_COLOR, rotation=30, fontsize=9)
    plt.setp(ax.get_yticklabels(), color=SUBTEXT_COLOR, fontsize=9)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, facecolor=PANEL_COLOR)
    buf.seek(0)
    chart_img = Image.open(buf)
    
    chart_img.load() 
    buf.close()

    plt.close(fig)
    
    return chart_img

def create_performance_chart(history: Dict) -> Optional[Image.Image]:
    if len(history['values']) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor=PANEL_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    
    samples_for_24h = 2880
    
    times = history['timestamps'][-samples_for_24h:]
    values = history['values'][-samples_for_24h:]
    
    if not values:
        return None 

    ax.plot(times, values, color=GOLD_COLOR, linewidth=2)
    ax.fill_between(times, values, alpha=0.3, color=GOLD_COLOR)
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        padding = min_val * 0.01
    else:
        padding = (max_val - min_val) * 0.1
        
    ax.set_ylim(min_val - padding, max_val + padding)
    # --------------------------------------------------
    
    ax.set_ylabel("Portfolio Value ($)", color=SUBTEXT_COLOR, fontsize=10)
    ax.tick_params(colors=SUBTEXT_COLOR, labelsize=9)
    ax.grid(True, alpha=0.1, color='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Sadece Saat:Dakika göster
    plt.setp(ax.get_xticklabels(), rotation=30)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, facecolor=PANEL_COLOR)
    buf.seek(0)
    perf_img = Image.open(buf)
    perf_img.load() 
    buf.close()     
    plt.close(fig)
    
    return perf_img
# =============================================================================
# ANIMATED DASHBOARD CREATION
# =============================================================================

def draw_panel(draw: ImageDraw, x: int, y: int, w: int, h: int, frame: int = 0):
    """Draw animated panel background"""
    panel_color = get_gradient_color(frame, PANEL_COLOR)
    draw.rectangle([x, y, x+w, y+h], fill=panel_color, outline='#30363d', width=1)

def draw_progress_bar(draw: ImageDraw, x: int, y: int, w: int, h: int, 
                     progress: float, color: str, bg_color: str = '#1c2128'):
    """Draw animated progress bar"""
    # Background
    draw.rectangle([x, y, x+w, y+h], fill=bg_color, outline='#30363d', width=1)
    # Progress
    fill_w = int(w * progress)
    if fill_w > 0:
        draw.rectangle([x, y, x+fill_w, y+h], fill=color)

def create_dashboard_image(portfolio_data: Dict[str, Any], frame: int):
    """Create fully animated dashboard"""
    global animation_frame
    animation_frame = frame
    
    img = Image.new('RGB', (1920, 1080), color=get_gradient_color(frame, BG_COLOR))
    draw = ImageDraw.Draw(img)

    # =========================
    # ANIMATED DISCLAIMER BAR
    # =========================
    disclaimer_bg = get_gradient_color(frame, PANEL_COLOR) # Arka planı panel ile aynı yaptık
    draw.rectangle([0, 0, 1920, 100], fill=disclaimer_bg)
    
    disclaimer_text = (
        "  EDUCATIONAL & RESEARCH PURPOSE ONLY - NOT FINANCIAL ADVICE    "
        "This is PAPER TRADING (simulated) with virtual money. Real trading involves substantial risk. "
        "This content is for educational purposes only. Not a recommendation to buy or sell."
    )
    
    words = disclaimer_text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        
        bbox = draw.textbbox((0, 0), test_line, font=FONT_SMALL) 
        
        if bbox[2] < 1850:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)
    
    dy = 20 
    for line in lines[:3]: 

        draw.text((40, dy), line.strip(), font=FONT_SMALL, fill=GOLD_COLOR)
        dy += 25 

    # =========================
    # HEADER WITH PULSING LIVE
    # =========================
    draw.text((40, 70), "CHIMERA", font=FONT_TITLE, fill=GOLD_COLOR)
    draw.text((40, 115), "Neuro-Symbolic Causal Trading Agent", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    
    pulse = get_pulse_alpha(frame, 0.1)
    live_color = (int(248 * pulse), int(81 * pulse), int(73 * pulse))
    
    draw.ellipse((1820, 50, 1840, 70), fill=live_color) 
    
    draw.text((1860, 55), "LIVE", font=FONT_HEADER, fill=TEXT_COLOR) 
    
    # -----------------------------------------------------------

    # Timestamp + Frame number
    time_text = f"{portfolio_data.get('timestamp', 'N/A')} | F:{frame}"
    draw.text((1860, 115), time_text, font=FONT_TINY, fill=SUBTEXT_COLOR, anchor="rt")

    # =========================
    # LEFT COLUMN - METRICS
    # =========================
    x, y = 40, 138
    
    # Portfolio Value
    draw_panel(draw, x, y, 400, 120, frame)
    draw.text((x+20, y+15), "Portfolio Value", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    draw.text((x+20, y+45), format_currency(portfolio_data.get('value', 0)), 
              font=FONT_TITLE, fill=TEXT_COLOR)
    
    y += 140
    
    # Position & P&L
    draw_panel(draw, x, y, 400, 180, frame)
    draw.text((x+20, y+15), "Current Position", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    draw.text((x+20, y+45), f"{portfolio_data.get('qty', 0):.6f} BTC", 
              font=FONT_HEADER, fill=BLUE_COLOR)
    
    pnl = portfolio_data.get('pnl', 0)
    pnl_pct = portfolio_data.get('pnl_pct', 0)
    pnl_color = GREEN_COLOR if pnl >= 0 else RED_COLOR
    
    draw.text((x+20, y+95), "Unrealized P&L", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    draw.text((x+20, y+120), f"{pnl:+,.2f}", font=FONT_MEDIUM, fill=pnl_color)
    draw.text((x+200, y+125), f"({pnl_pct:+.2f}%)", font=FONT_SMALL, fill=pnl_color)
    
    y += 200
    
    # =========================
    # Performance Metrics
    # =========================
    panel_height = 180 
    draw_panel(draw, x, y, 400, panel_height, frame)
    draw.text((x+20, y+15), "Performance Metrics", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    
    # --- Total P&L ---
    dy = y + 50 
    total_pnl = portfolio_data.get('total_pnl', 0.0)
    total_pnl_pct = portfolio_data.get('total_pnl_pct', 0.0)
    pnl_color = GREEN_COLOR if total_pnl >= 0 else RED_COLOR
    
    draw.text((x+20, dy), "Total P&L:", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    draw.text((x+160, dy), f"{total_pnl:+,.2f}", font=FONT_MEDIUM, fill=pnl_color)
    draw.text((x+280, dy+5), f"({total_pnl_pct:+.2f}%)", font=FONT_SMALL, fill=pnl_color)
    
    # --- Sharpe Ratio ---
    dy += 40 # 40px aşağı kay
    sharpe = portfolio_data.get('sharpe', 0.0)
    sharpe_color = GREEN_COLOR if sharpe > 1.0 else GOLD_COLOR if sharpe > 0 else RED_COLOR
    
    draw.text((x+20, dy), "Sharpe Ratio:", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    draw.text((x+160, dy), f"{sharpe:.2f}", font=FONT_MEDIUM, fill=sharpe_color)
    
    # --- Max Drawdown ---
    dy += 40 
    max_dd = portfolio_data.get('max_dd', 0.0)
    dd_color = GREEN_COLOR if max_dd > -5 else GOLD_COLOR if max_dd > -15 else RED_COLOR
    
    draw.text((x+20, dy), "Max Drawdown:", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    draw.text((x+160, dy), f"{max_dd:.1f}%", font=FONT_MEDIUM, fill=dd_color)
    
    y += panel_height + 20 
    
    # =========================
    # ANIMATED COUNTDOWN TIMER
    # =========================
    draw_panel(draw, x, y, 400, 160, frame)
    draw.text((x+20, y+15), "Next Agent Run In", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    
    countdown = get_next_run_countdown()
    timer_text = f"{countdown['hours']:02d}:{countdown['minutes']:02d}:{countdown['seconds']:02d}"
    draw.text((x+20, y+50), timer_text, font=FONT_TITLE, fill=PURPLE_COLOR)
    
    # Animated progress bar
    total_day_seconds = 86400
    progress = 1 - (countdown['total_seconds'] / total_day_seconds)
    draw_progress_bar(draw, x+20, y+110, 360, 20, progress, PURPLE_COLOR)
    
    percent_text = f"{progress*100:.1f}% of day elapsed"
    draw.text((x+200, y+115), percent_text, font=FONT_TINY, fill=SUBTEXT_COLOR, anchor="mm")


    y += 180  # 160px timer paneli + 20px boşluk
    
    draw_panel(draw, x, y, 400, 130, frame) # 400 genişlik, 130 yükseklik
    draw.text((x+20, y+15), "Support This Project", font=FONT_MEDIUM, fill=TEXT_COLOR)
    
    # GitHub Satırı
    draw.text((x+20, y+55), "Star on GitHub", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    draw.text((x+380, y+55), "Project-Chimera", font=FONT_REGULAR, fill=GOLD_COLOR, anchor="rt")
    
    # Medium Satırı
    draw.text((x+20, y+90), "Follow on Medium", font=FONT_REGULAR, fill=SUBTEXT_COLOR)
    draw.text((x+380, y+90), "@akarlaraytu", font=FONT_REGULAR, fill=GOLD_COLOR, anchor="rt")
    # =========================

    # =========================
    # CENTER - CHARTS
    # =========================
    x = 480
    y = 210
    
    # BTC Price Chart
    draw_panel(draw, x, y, 740, 380, frame)
    draw.text((x+20, y+15), f"BTC/USD - Last 12 Hours", 
              font=FONT_MEDIUM, fill=TEXT_COLOR)
    
    chart_img = portfolio_data.get('chart_img')
    if chart_img:
        chart_img = chart_img.resize((720, 340))
        img.paste(chart_img, (x+10, y+40))
    
    y += 400
    
    # Portfolio Performance Chart
    draw_panel(draw, x, y, 740, 280, frame)
    draw.text((x+20, y+15), "Portfolio Value (Last 24 Hours)", 
              font=FONT_MEDIUM, fill=TEXT_COLOR)
    
    perf_img = portfolio_data.get('perf_img')
    if perf_img:
        perf_img = perf_img.resize((720, 240))
        img.paste(perf_img, (x+10, y+40))
    
    # =========================
    # RIGHT - ACTIVITY
    # =========================
    x = 1260
    y = 210
    
    # Trade History
    draw_panel(draw, x, y, 620, 300, frame)
    draw.text((x+20, y+15), "Recent Trades", font=FONT_MEDIUM, fill=TEXT_COLOR)
    
    trades = portfolio_data.get('trade_history', [])
    ty = y + 55
    for trade in trades[:6]:
        side = trade['side'].upper()
        side_color = GREEN_COLOR if side == 'BUY' else RED_COLOR
        
        timestamp = pd.to_datetime(trade['timestamp']).strftime('%m/%d %H:%M')
        text = f"[{timestamp}] {side} {trade['qty']:.4f} @ ${trade['price']:,.0f}"
        
        draw.text((x+20, ty), text, font=FONT_SMALL, fill=side_color)
        ty += 40
    
    if not trades:
        draw.text((x+20, ty), "No trades yet", font=FONT_SMALL, fill=SUBTEXT_COLOR)
    
    y += 320
    
    # Agent Decision
    draw_panel(draw, x, y, 620, 360, frame)
    draw.text((x+20, y+15), "Last Agent Analysis", font=FONT_MEDIUM, fill=TEXT_COLOR)
    
    agent_data = portfolio_data.get('agent_analysis')
    if agent_data:
        action = agent_data.get('final_action', {})
        commentary = agent_data.get('commentary', 'N/A')[:150]
        
        draw.text((x+20, y+50), "Decision:", font=FONT_SMALL, fill=SUBTEXT_COLOR)
        action_text = f"{action.get('type', 'HOLD')} {action.get('amount', 0):.1%}"
        action_color = GREEN_COLOR if action.get('type') == 'BUY' else RED_COLOR if action.get('type') in ['SELL', 'SHORT'] else GOLD_COLOR
        draw.text((x+120, y+50), action_text, font=FONT_MEDIUM, fill=action_color)
        
        draw.text((x+20, y+85), "Reasoning:", font=FONT_SMALL, fill=SUBTEXT_COLOR)
        words = commentary.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + word) < 45:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        cy = y + 115
        for line in lines[:3]:
            draw.text((x+20, cy), line, font=FONT_SMALL, fill=TEXT_COLOR)
            cy += 25
        
        scenarios = agent_data.get('scenarios', [])
        sy = y + 210
        draw.text((x+20, sy), "Tested Scenarios:", font=FONT_SMALL, fill=SUBTEXT_COLOR)
        sy += 30
        
        for scenario in scenarios[:4]:
            hypo = scenario.get('hypothesis', 'N/A')[:20]
            validation = scenario.get('validation', 'Pending')
            impact = scenario.get('impact')
            
            status = "[OK]" if validation == "Valid" else "[X]"
            status_color = GREEN_COLOR if validation == "Valid" else RED_COLOR
            
            draw.text((x+20, sy), f"{status} {hypo}", font=FONT_TINY, fill=status_color)
            
            if impact is not None and validation == "Valid":
                impact_text = f"{impact:+.1%}"
                impact_color = GREEN_COLOR if impact > 0 else RED_COLOR
                draw.text((x+320, sy), impact_text, font=FONT_TINY, fill=impact_color)
            
            sy += 25
    else:
        draw.text((x+20, y+50), "Waiting for first agent cycle...", 
                  font=FONT_SMALL, fill=SUBTEXT_COLOR)
    
    # =========================
    # FOOTER - SCROLLING LOG
    # =========================
    y = 1000
    draw.line((40, y, 1880, y), fill='#30363d', width=1)
    
    logs = portfolio_data.get('recent_logs', [])
    # Animate log scroll
    scroll_offset = (frame // 5) % len(logs) if logs else 0
    scrolled_logs = logs[scroll_offset:] + logs[:scroll_offset]
    
    log_text = " | ".join(scrolled_logs[-3:]) if scrolled_logs else "System ready"
    draw.text((40, y+10), log_text[:180], font=FONT_TINY, fill=SUBTEXT_COLOR)
    
    frame_filename = f"dashboard_frame_{frame:06d}.png"
    output_path = os.path.join(FRAMES_DIR, frame_filename)
    temp_path = output_path + ".tmp"
    
    try:
        img.save(temp_path, "PNG")
        os.rename(temp_path, output_path)
    except Exception as e:
        print(f"⚠️  Frame save error: {e}")
        # Fallback
        img.save(output_path, "PNG")
        
        
    frame_to_delete = frame - FRAME_BUFFER_SIZE
    if frame_to_delete >= 0:
        try:
            old_frame_filename = f"dashboard_frame_{frame_to_delete:06d}.png"
            old_frame_path = os.path.join(FRAMES_DIR, old_frame_filename)
            os.remove(old_frame_path)
        except FileNotFoundError:
            # Silinecek dosya zaten yoksa sorun değil, devam et.
            pass
        except Exception as e:
            print(f"⚠️  Could not delete old frame {old_frame_path}: {e}")
    # =========================================================
    
    return img


# =============================================================================
# DATA FETCHER THREAD
# =============================================================================

def data_fetcher_loop():
    """
    Arka planda çalışan, veriyi çeken VE ana thread'i izleyen 'Watchdog'.
    """
    global g_cached_data, g_data_lock, g_last_frame_heartbeat
    
    print("🛰️  Data fetcher thread (and Watchdog) started...")
    
    while True:
        try:
            # --- 1. WATCHDOG KONTROLÜ ---
            time_since_last_heartbeat = time.time() - g_last_frame_heartbeat
            
            # Ana thread 90 saniyedir (3x data fetch aralığı) donmuşsa,
            # tüm programı yeniden başlat.
            if time_since_last_heartbeat > 90.0:
                print("="*60)
                print("🚨 WATCHDOG: Main render thread is frozen!")
                print(f"🚨 Last heartbeat was {time_since_last_heartbeat:.0f} seconds ago.")
                print("🚨 Forcing script restart...")
                print("="*60)
                
                # Bu, tüm prosesi öldürür.
                # 'tmux' veya 'systemd' gibi bir servis yöneticisi
                # script'i otomatik olarak yeniden başlatmalıdır.
                os._exit(1) # 1 "hata" koduyla çık
                
            # --- 2. VERİ ÇEKME İŞİ ---
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Fetching new data in background...")
            new_data = fetch_all_data()
            
            with g_data_lock:
                g_cached_data = new_data
                
            print("✅ New data fetched and cached.")
            
        except Exception as e:
            print(f"❌ Error in data fetcher thread: {e}")
            
        time.sleep(DATA_FETCH_INTERVAL)

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_all_data() -> Dict[str, Any]:
    """Fetch all data for dashboard"""
    try:
        account = api.get_account()
        
        qty, pnl, pnl_pct = 0.0, 0.0, 0.0
        try:
            position = api.get_position(SYMBOL_DISPLAY)
            qty = float(position.qty)
            pnl = float(position.unrealized_pl)
            pnl_pct = float(position.unrealized_plpc) * 100
        except APIError as e:
            if "position not found" not in str(e).lower():
                print(f"⚠️  Position error: {e}")
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=12)
        
        market_data_df = api.get_crypto_bars(
            SYMBOL_ALPACA,
            '5Min',
            start=start_date.isoformat(), 
            end=end_date.isoformat()  
        ).df
        
        market_data_df.rename(columns={
            'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close', 
            'volume': 'Volume'
        }, inplace=True)
        
        portfolio_value = float(account.equity)
        history = update_performance_history(portfolio_value)
        total_pnl = portfolio_value - STARTING_CAPITAL
        total_pnl_pct = (total_pnl / STARTING_CAPITAL) * 100
        
        sharpe = 0.0
        max_dd = 0.0
        if len(history['daily_returns']) > 7:
            sharpe = calculate_sharpe_ratio(history['daily_returns'])
            max_dd = calculate_max_drawdown(history['values'])
        
        orders = api.list_orders(status='closed', limit=20, direction='desc')
        trade_history = update_trade_history(orders)
        
        agent_analysis = load_agent_analysis()
        recent_logs = read_recent_logs()
        
        chart_img = create_btc_chart_image(market_data_df)
        perf_img = create_performance_chart(history)
        
        return {
            'value': portfolio_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'qty': qty,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trade_history': trade_history,
            'agent_analysis': agent_analysis,
            'recent_logs': recent_logs,
            'total_pnl': total_pnl,    
            'total_pnl_pct': total_pnl_pct, 
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'chart_days': 0.5, 
            'chart_img': chart_img,
            'perf_img': perf_img
        }
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'value': 0, 'pnl': 0, 'pnl_pct': 0, 'qty': 0,
            'sharpe': 0.0, 'max_dd': 0.0,
            'trade_history': [],
            'agent_analysis': None,
            'recent_logs': [f"Error: {str(e)}"],
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'chart_days': 0,
            'chart_img': None,
            'perf_img': None
        }

# =============================================================================
# YOUTUBE STREAMING
# =============================================================================

class YouTubeStreamer:
    """Handles YouTube streaming via FFmpeg with pipe"""
    
    def __init__(self):
        self.process = None
        self.stream_key = os.getenv("YOUTUBE_STREAM_KEY", "")
        self.rtmp_url = "rtmp://a.rtmp.youtube.com/live2/"
        self.current_frame = 0
        
    def start(self):
        """Start FFmpeg stream with stdin pipe"""
        if not self.stream_key:
            print("❌ YOUTUBE_STREAM_KEY not set")
            return False
        
        stream_url = f"{self.rtmp_url}{self.stream_key}"
        
        # Use PIPE input instead of file sequence
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'image2pipe',
            '-framerate', '1',
            '-i', '-',  # Read from stdin
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-s', '1920x1080',
            '-r', '30',
            '-g', '60',
            '-b:v', '5000k',
            '-maxrate', '5000k',
            '-minrate', '5000k',
            '-bufsize', '10000k',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            '-f', 'flv',
            stream_url
        ]
        
        print("📺 Starting YouTube stream with pipe...")
        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            print("✅ Stream started!")
            return True
        except Exception as e:
            print(f"❌ Stream failed: {e}")
            return False
    
    def send_frame(self, frame_number: int):
        """Send a frame to FFmpeg via pipe. Returns False on failure."""
        if not self.process or not self.is_running():
            return False
        
        frame_path = os.path.join(FRAMES_DIR, f"dashboard_frame_{frame_number:06d}.png")
        
        if os.path.exists(frame_path):
            try:
                with open(frame_path, 'rb') as f:
                    self.process.stdin.write(f.read())
                    self.process.stdin.flush()
                return True
            except (BrokenPipeError, OSError) as e:
                print(f"⚠️  Stream pipe error: {e}. Stream will restart.")
                try:
                    self.process.terminate() # Ölmesini garantile
                except:
                    pass
                return False
            except Exception as e:
                print(f"⚠️  Unknown error in send_frame: {e}")
                return False

        return False
    
    def is_running(self) -> bool:
        if self.process:
            return self.process.poll() is None
        return False
    
    def stop(self):
        if self.process:
            print("\n🛑 Stopping stream...")
            try:
                self.process.stdin.close()
            except:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("✅ Stream stopped")

# =============================================================================
# MAIN LOOP
# =============================================================================

def main(stream_mode: bool = False):
    """Main dashboard loop with animation"""
    print("="*60)
    print("  CHIMERA ANIMATED DASHBOARD")
    print("="*60)
    print(f"  Mode: {'STREAMING' if stream_mode else 'LOCAL ONLY'}")
    print(f"  Render Interval: {UPDATE_INTERVAL_SECONDS}s")
    print(f"  Data Fetch Interval: {DATA_FETCH_INTERVAL}s")
    print("="*60)
    print("")
    
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for old_file in os.listdir(FRAMES_DIR):
        if old_file.startswith("dashboard_frame_"):
            os.remove(os.path.join(FRAMES_DIR, old_file))
    print(f"🖼️  Frames will be saved to: {FRAMES_DIR}")
        
    global g_cached_data, g_data_lock, g_last_frame_heartbeat
    frame = 0
    last_render = 0
    
    try:
        # --- İLK VERİ ÇEKME ---
        print("Fetching initial data...")
        g_cached_data = fetch_all_data()
        if not g_cached_data:
            print("❌ Failed to fetch initial data. Exiting.")
            return
        print("✅ Initial data loaded")
        
        # --- VERİ ÇEKME THREAD'İNİ BAŞLAT ---
        fetcher_thread = threading.Thread(target=data_fetcher_loop, daemon=True)
        fetcher_thread.start()
        
        # --- İLK FRAMELERİ OLUŞTUR ---
        print("Generating initial frames for stream...")
        for i in range(5):
            # Kilitli alanda sadece veriyi kopyala
            local_data = None
            with g_data_lock:
                local_data = g_cached_data.copy()
                
            local_data['timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            create_dashboard_image(local_data, i)
            
            frame = i
            time.sleep(0.2)
        frame += 1
        print(f"✅ {frame+1} initial frames generated")
        print("")
        
        streamer = None
        if stream_mode:
            streamer = YouTubeStreamer()
            if not streamer.start():
                print("⚠️  Continuing without streaming...")
                streamer = None
        
        while True:
            current_time = time.time()
            
            if current_time - last_render >= UPDATE_INTERVAL_SECONDS:
                frame += 1
                
                local_data_copy = None
                with g_data_lock:
                    if g_cached_data:
                        local_data_copy = g_cached_data.copy()
                
                if local_data_copy:
                    local_data_copy['timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                    try:
                        # 1. Render et ve diske kaydet
                        create_dashboard_image(local_data_copy, frame)
                        
                        if streamer:
                            
                            # --- 2. ÇÖZÜM: PROAKTİF RESET ---
                            # 5 dakikalık donmayı önlemek için, 4 dakikada bir (240 frame)
                            # ffmpeg prosesini biz yeniden başlatalım.
                            if frame % 240 == 0 and frame > 0:
                                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] --- Proactive Stream Restart @ Frame {frame} ---")
                                streamer.stop()
                                time.sleep(1) # Kapanması için 1sn bekle
                                streamer.start()
                                time.sleep(2) # Bağlanması için 2sn bekle
                            # ---------------------------------

                            # 3. Stream'i kontrol et
                            if not streamer.is_running():
                                print("⚠️  Stream process found dead (post-check). Restarting...")
                                streamer.start()
                            else:
                                # 4. Frame'i gönder
                                send_success = streamer.send_frame(frame)
                                
                                # 5. KALP ATIŞINI GÜNCELLE (Watchdog için)
                                g_last_frame_heartbeat = time.time()
                                
                                if send_success and (frame % 30 == 0):
                                    timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
                                    print(f"[{timestamp}] Frame #{frame} sent to stream ✨")
                                elif not send_success:
                                    print(f"[{timestamp}] Frame #{frame} failed to send (pipe error).")

                    except Exception as e:
                        print(f"❌ Render failed: {e}")
                
                last_render = current_time
            
            # 5. Uyu
            sleep_time = max(0, UPDATE_INTERVAL_SECONDS - (time.time() - last_render))
            time.sleep(sleep_time * 0.5)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutdown signal received...")
        if streamer:
            streamer.stop()
        print("✅ Dashboard stopped cleanly")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chimera Animated Dashboard')
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Enable YouTube streaming (requires YOUTUBE_STREAM_KEY in .env)'
    )
    
    args = parser.parse_args()
    
    # Check API keys
    if not API_KEY or not SECRET_KEY:
        print("❌ FATAL: Alpaca API keys not found in .env")
        sys.exit(1)
    
    if args.stream and not os.getenv("YOUTUBE_STREAM_KEY"):
        print("⚠️  WARNING: --stream enabled but YOUTUBE_STREAM_KEY not set")
        print("   Add to .env: YOUTUBE_STREAM_KEY=your-key-here")
        response = input("Continue without streaming? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        args.stream = False
    
    main(stream_mode=args.stream)