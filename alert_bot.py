# NSE alert bot v6 - dynamic watchlist + BUY alerts ranked by quick-rise strength.
# Runs on GitHub Actions every 5 minutes (see scan.yml).
#
# At ~10:00 AM and ~1:00 PM IST it scans the whole universe (Nifty 500), keeps the
# 47 liquid stocks with the strongest momentum, and uses that shortlist until the
# next refresh. Before 10:00 AM it uses the default 47-stock list.
#
# Env var (GitHub secret): NTFY_TOPIC
# Optional: MIN_QUICK (default 5), MAX_ALERTS (10), SHORTLIST_SIZE (47),
#           MIN_TURNOVER_CR (10 = Rs 10 crore average daily traded value)

import os
import io
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = "state.json"
WATCHLIST_FILE = "watchlist.json"

MIN_QUICK = float(os.getenv("MIN_QUICK", "5"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "47"))
MIN_TURNOVER_CR = float(os.getenv("MIN_TURNOVER_CR", "10"))
MIN_PRICE = 30

UNIVERSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC",
    "SBIN", "LT", "HINDUNILVR", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
    "MARUTI", "SUNPHARMA", "HCLTECH", "M&M", "NTPC", "TITAN", "ULTRACEMCO",
    "ONGC", "POWERGRID", "ADANIPORTS", "WIPRO", "COALINDIA", "JSWSTEEL",
    "TATASTEEL", "BAJAJFINSV", "NESTLEIND", "ASIANPAINT", "TECHM", "GRASIM",
    "HINDALCO", "CIPLA", "DRREDDY", "EICHERMOT", "APOLLOHOSP", "BPCL",
    "HEROMOTOCO", "SBILIFE", "HDFCLIFE", "INDUSINDBK", "TRENT",
    "SHRIRAMFIN", "BAJAJ-AUTO", "ADANIENT", "BEL",
]


# ---------------------------------------------------------------- indicators

def tick(x):
    return round(round(x / 0.05) * 0.05, 2)


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))


def analyse_df(symbol, df):
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in cols):
        return None
    df = df[cols].dropna().copy()
    if df.empty:
        return None

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

    # Average daily traded value on previous days (Rs crore) - liquidity filter
    turn = [float((df[df["date"] == d]["Close"] * df[df["date"] == d]["Volume"]).sum())
            for d in days[:-1]]
    turnover_cr = (sum(turn) / len(turn)) / 1e7 if turn else 0.0

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
            "target": target, "why": "; ".join(why),
            "turnover_cr": round(turnover_cr, 1),
            "last_date": last_ts.date().isoformat()}


# ---------------------------------------------------------------- data fetching

def get_universe():
    """All Nifty 500 symbols from NSE. Falls back to the default list if the download fails."""
    try:
        resp = requests.get(UNIVERSE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        table = pd.read_csv(io.StringIO(resp.text))
        col = [c for c in table.columns if c.strip().lower() == "symbol"][0]
        symbols = sorted(set(table[col].astype(str).str.strip()))
        if len(symbols) > 100:
            print("Universe loaded:", len(symbols), "stocks")
            return symbols
    except Exception as e:
        print("Could not download the stock universe:", e)
    print("Using the default watchlist as the universe.")
    return list(DEFAULT_WATCHLIST)


def fetch_batch(symbols):
    """Download 5-minute data for several stocks at once. Returns {symbol: DataFrame}."""
    out = {}
    tickers = [s + ".NS" for s in symbols]
    try:
        data = yf.download(tickers, period="5d", interval="5m", group_by="ticker",
                           threads=True, progress=False, auto_adjust=True)
    except Exception as e:
        print("Batch download failed:", e)
        return out
    if data is None or data.empty:
        return out
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        for s in symbols:
            tk = s + ".NS"
            if tk in level0:
                sub = data[tk].dropna(how="all")
                if not sub.empty:
                    out[s] = sub
    elif len(symbols) == 1:
        out[symbols[0]] = data.dropna(how="all")
    return out


def analyse_many(symbols, batch_size=50):
    results = []
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        frames = fetch_batch(chunk)
        if not frames:                      # one retry if Yahoo returned nothing
            time.sleep(5)
            frames = fetch_batch(chunk)
        for s, d in frames.items():
            try:
                res = analyse_df(s, d)
            except Exception as e:
                print("skipped", s, e)
                continue
            if res:
                results.append(res)
        time.sleep(1)
    return results


def pick_shortlist(results):
    """Liquid stocks traded in the latest session, ranked by quick-rise strength."""
    if not results:
        return []
    latest = max(r["last_date"] for r in results)
    ok = [r for r in results
          if r["last_date"] == latest
          and r["turnover_cr"] >= MIN_TURNOVER_CR
          and r["price"] >= MIN_PRICE]
    ok.sort(key=lambda r: r["quick"], reverse=True)
    return ok[:SHORTLIST_SIZE]


# ---------------------------------------------------------------- notifications

def notify(text):
    """Send a push notification to the ntfy app. Returns True if it was delivered."""
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC secret missing. Message was:\n", text)
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


# ---------------------------------------------------------------- saved state

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


def load_watchlist(today):
    try:
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
        if data.get("date") == today and data.get("stocks"):
            return data
    except Exception:
        pass
    return {"date": today, "refreshed": [], "stocks": list(DEFAULT_WATCHLIST)}


def save_watchlist(wl):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f)


# ---------------------------------------------------------------- main

def main():
    if os.getenv("TEST", "").lower() == "true" or "test" in sys.argv:
        ok = notify("Scanner connected. You will get BUY alerts here "
                           "during NSE market hours.")
        print("Test message sent." if ok else "Test message failed.")
        return

    now = datetime.now(IST)
    hm = (now.hour, now.minute)
    force_refresh = os.getenv("FORCE_REFRESH") == "1"
    in_hours = now.weekday() < 5 and (9, 30) <= hm <= (15, 25)
    if not in_hours and os.getenv("FORCE") != "1" and not force_refresh:
        print("Outside market hours, nothing to do.", now)
        return

    today = now.date().isoformat()
    state = load_state(today)
    wl = load_watchlist(today)

    # Decide whether the watchlist should be refreshed on this run
    slot = None
    if force_refresh:
        slot = "M"
    elif (10, 0) <= hm < (12, 30) and "A" not in wl["refreshed"]:
        slot = "A"
    elif (13, 0) <= hm < (14, 30) and "B" not in wl["refreshed"]:
        slot = "B"

    if slot:
        universe = get_universe()
        all_results = analyse_many(universe)
        picks = pick_shortlist(all_results)
        wl["refreshed"].append(slot)
        if len(picks) >= 10:
            wl["stocks"] = [r["stock"] for r in picks]
            results = picks
            names = ", ".join(r["stock"] for r in picks[:15])
            notify(f"Watchlist refreshed {now.strftime('%H:%M')} IST. "
                          f"Scanned {len(all_results)} of {len(universe)} stocks, "
                          f"shortlisted {len(picks)} with the strongest momentum.\n"
                          f"Top: {names}")
        else:
            print("Shortlist too small; keeping the current watchlist.")
            notify(f"Watchlist refresh at {now.strftime('%H:%M')} IST failed "
                          f"(only {len(all_results)} stocks had data). Keeping the previous list.")
            results = analyse_many(wl["stocks"])
        save_watchlist(wl)
    else:
        results = analyse_many(wl["stocks"])

    candidates = []
    for res in results:
        # Alert only on actionable buys with strong quick-rise strength
        if res["action"].startswith(("BUY NOW", "WATCH")) and res["quick"] >= MIN_QUICK:
            key = f"{res['stock']}:BUY"
            prev = state.get(key)
            # New today, or upgraded from STRONG to HOT
            if prev is None or (str(prev).endswith("STRONG") and res["rise"] == "HOT"):
                candidates.append((key, res))

    if not candidates:
        print("No new buy signals.", now.strftime("%H:%M"), "stocks scanned:", len(results))
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

    if notify("\n".join(lines)):
        for key, res in candidates:
            state[key] = f"{today}|{res['rise']}"
    save_state(state)


if __name__ == "__main__":
    main()
