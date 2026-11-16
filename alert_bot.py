# NSE alert bot v5 - BUY alerts only, ranked by quick-rise strength.
# Runs on GitHub Actions every 5 minutes (see scan.yml).
# Env vars: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID  (set as GitHub secrets)
# Optional: MIN_QUICK (default 5 = STRONG or HOT), MAX_ALERTS (default 10)

import os
import sys
import json
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = "state.json"
MIN_QUICK = float(os.getenv("MIN_QUICK", "5"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))

WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC",
    "SBIN", "LT", "HINDUNILVR", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
    "MARUTI", "SUNPHARMA", "HCLTECH", "M&M", "NTPC", "TITAN", "ULTRACEMCO",
    "ONGC", "POWERGRID", "ADANIPORTS", "WIPRO", "COALINDIA", "JSWSTEEL",
    "TATASTEEL", "BAJAJFINSV", "NESTLEIND", "ASIANPAINT", "TECHM", "GRASIM",
    "HINDALCO", "CIPLA", "DRREDDY", "EICHERMOT", "APOLLOHOSP", "BPCL",
    "HEROMOTOCO", "SBILIFE", "HDFCLIFE", "INDUSINDBK", "TRENT",
    "SHRIRAMFIN", "BAJAJ-AUTO", "ADANIENT", "BEL",
]


def tick(x):
    return round(round(x / 0.05) * 0.05, 2)


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))


def analyse(symbol):
    df = yf.download(symbol + ".NS", period="5d", interval="5m",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(IST)
    df["date"] = df.index.date

    days = sorted(df["date"].unique())
    if len(days) < 2:
        return None

    today = days[-1]
    t = df[df["date"] == today]
    n = len(t)
    prev_close = float(df[df["date"] == days[-2]]["Close"].iloc[-1])

    last = float(t["Close"].iloc[-1])
    day_open = float(t["Open"].iloc[0])
    chg = (last / prev_close - 1) * 100
    from_open = (last / day_open - 1) * 100

    ref = float(t["Close"].iloc[-7]) if n >= 7 else day_open
    mom30 = (last / ref - 1) * 100

    tp = (t["High"] + t["Low"] + t["Close"]) / 3
    cum_vol = t["Volume"].cumsum().iloc[-1]
    vwap = float((tp * t["Volume"]).cumsum().iloc[-1] / cum_vol) if cum_vol > 0 else last

    prev_vols = [df[df["date"] == d]["Volume"].iloc[:n].sum() for d in days[:-1]]
    avg_prev = sum(prev_vols) / len(prev_vols) if prev_vols else 0
    vol_ratio = (t["Volume"].sum() / avg_prev) if avg_prev > 0 else 1.0

    close = df["Close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    diff = ema9 - ema21
    r = float(rsi(close).iloc[-1])

    prev_c = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_c).abs(),
                    (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])

    # ---------- Base score ----------
    score, why = 0.0, []
    brk_up = False

    if last > vwap:
        score += 1; why.append("above VWAP")
    else:
        score -= 1; why.append("below VWAP")

    if from_open > 1:
        score += 1; why.append(f"+{from_open:.1f}% from open")
    elif from_open < -1:
        score -= 1; why.append(f"{from_open:.1f}% from open")

    if n >= 4:
        orb_high = float(t["High"].iloc[:3].max())
        orb_low = float(t["Low"].iloc[:3].min())
        if last > orb_high:
            brk_up = True; score += 1.5; why.append("opening-range breakout")
        elif last < orb_low:
            score -= 1.5; why.append("opening-range breakdown")

    crossed_up = (diff.iloc[-4:-1] < 0).any() and diff.iloc[-1] > 0
    crossed_dn = (diff.iloc[-4:-1] > 0).any() and diff.iloc[-1] < 0
    if crossed_up:
        score += 1.5; why.append("fresh EMA bullish cross")
    elif crossed_dn:
        score -= 1.5; why.append("fresh EMA bearish cross")
    elif diff.iloc[-1] > 0:
        score += 0.5; why.append("short-term up")
    else:
        score -= 0.5; why.append("short-term down")

    if vol_ratio > 1.5:
        score += (1 if chg > 0 else -1)
        why.append(f"volume {vol_ratio:.1f}x normal")

    if r > 75:
        score -= 0.5; why.append(f"RSI {r:.0f} stretched")
    elif r < 25:
        score += 0.5; why.append(f"RSI {r:.0f} oversold")

    # ---------- Quick-rise strength ----------
    ext = abs(last - vwap) / atr if atr > 0 else 0
    extended = ext > 5 or r > 80 or from_open > 3
    quick = score + max(-1.0, min(mom30, 2.0)) + 0.5 * max(0.0, min(vol_ratio - 1, 3.0))
    if extended:
        quick -= 2
        why.append("already extended")
    if mom30 > 0.3:
        why.append(f"+{mom30:.1f}% last 30 min")

    if quick >= 7:
        rise = "HOT"
    elif quick >= 5:
        rise = "STRONG"
    elif quick >= 3:
        rise = "MODERATE"
    else:
        rise = "WEAK"

    # ---------- Entry timing (buy side only) ----------
    last_ts = t.index[-1]
    now = datetime.now(IST)
    mins_from_open = last_ts.hour * 60 + last_ts.minute - (9 * 60 + 15)
    market_closed = (now.weekday() >= 5 or last_ts.date() != now.date()
                     or (now.hour, now.minute) >= (15, 35)
                     or (now.hour, now.minute) < (9, 15))
    too_early = mins_from_open < 30
    too_late = (last_ts.hour, last_ts.minute) >= (14, 45)
    confirmed = vol_ratio >= 1.5 and brk_up and diff.iloc[-1] > 0

    risk = 2.5 * atr
    action, entry, stop, target = "NO TRADE", "-", "-", "-"

    if score >= 3:
        if market_closed:
            action = "BUY setup (market closed)"
        elif too_early:
            action = "WAIT - too early"
        elif too_late:
            action = "AVOID - too late for fresh entry"
        elif extended:
            action = f"WAIT - extended, buy on dip near {tick(float(ema9.iloc[-1]))}"
        elif confirmed:
            action = "BUY NOW (confirmed)"
        else:
            action = "WATCH - not confirmed"
        if action.startswith(("BUY NOW", "WATCH")):
            entry = tick(last)
            stop, target = tick(last - risk), tick(last + 1.5 * risk)
    elif score >= 1.5:
        action = "EARLY - building, wait for confirmation"

    return {"stock": symbol, "price": round(last, 2), "mom30": round(mom30, 2),
            "vol_x": round(vol_ratio, 1), "score": score, "quick": round(quick, 1),
            "rise": rise, "action": action, "entry": entry, "stop": stop,
            "target": target, "why": "; ".join(why)}


def send_ntfy(text):
    """Push notification to the ntfy app on the iPhone (topic = NTFY_TOPIC secret)."""
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        return False
    try:
        resp = requests.post(f"https://ntfy.sh/{topic}",
                             data=text[:3500].encode("utf-8"),
                             headers={"Title": "NSE BUY alert", "Priority": "high"},
                             timeout=20)
        print("ntfy status:", resp.status_code)
        return resp.ok
    except Exception as e:
        print("ntfy error:", e)
        return False


def send_telegram(text):
    """Sends via ntfy and/or Telegram. Returns True if at least one worked."""
    ok_ntfy = send_ntfy(text)

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    ok_tg = False
    if token and chat_id:
        try:
            resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data={"chat_id": chat_id, "text": text[:4000]}, timeout=20)
            print("Telegram status:", resp.status_code, resp.text[:200])
            ok_tg = resp.ok
        except Exception as e:
            print("Telegram error:", e)
    elif not os.getenv("NTFY_TOPIC"):
        print("No notification channel configured. Message was:\n", text)
    return ok_ntfy or ok_tg


def load_state(today):
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if str(v).startswith(today)}
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def main():
    if os.getenv("TEST", "").lower() == "true" or "test" in sys.argv:
        ok = send_telegram("Scanner connected. You will get BUY alerts here "
                           "during NSE market hours.")
        print("Test message sent." if ok else "Test message failed.")
        return

    now = datetime.now(IST)
    in_hours = now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) <= (15, 25)
    if not in_hours and os.getenv("FORCE") != "1":
        print("Outside market hours, nothing to do.", now)
        return

    today = now.date().isoformat()
    state = load_state(today)
    candidates = []

    for s in WATCHLIST:
        try:
            res = analyse(s)
        except Exception as e:
            print("skipped", s, e)
            continue
        if not res:
            continue
        # Alert only on actionable buys with strong quick-rise strength
        if res["action"].startswith(("BUY NOW", "WATCH")) and res["quick"] >= MIN_QUICK:
            key = f"{s}:BUY"
            prev = state.get(key)
            # New today, or upgraded from STRONG to HOT
            if prev is None or (str(prev).endswith("STRONG") and res["rise"] == "HOT"):
                candidates.append((key, res))

    if not candidates:
        print("No new buy signals.", now.strftime("%H:%M"))
        save_state(state)
        return

    candidates.sort(key=lambda x: x[1]["quick"], reverse=True)
    candidates = candidates[:MAX_ALERTS]

    lines = [f"NSE BUY candidates {now.strftime('%H:%M')} IST (fastest rise first)", ""]
    for i, (key, res) in enumerate(candidates, 1):
        lines.append(f"{i}. {res['stock']} @ {res['entry']}  [{res['rise']}, quick {res['quick']}]")
        lines.append(f"   {res['action']}")
        lines.append(f"   Stop {res['stop']} | Target {res['target']}")
        lines.append(f"   30-min {res['mom30']:+.1f}% | volume {res['vol_x']}x")
        lines.append(f"   {res['why']}")
        lines.append("")
    lines.append("Ranking by current momentum, not a guaranteed rise. Not financial advice.")
    lines.append("Always use the stop-loss.")

    if send_telegram("\n".join(lines)):
        for key, res in candidates:
            state[key] = f"{today}|{res['rise']}"
    save_state(state)


if __name__ == "__main__":
    main()
