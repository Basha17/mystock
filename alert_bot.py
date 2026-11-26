# NSE alert bot v7 - dynamic watchlist, CONFIRMED/WATCH/EARLY BUY alerts, outcome log + daily summary.
# Runs on GitHub Actions every 5 minutes (see scan.yml).
#
# Every hour from 9:30 AM to 1:30 PM IST it scans the whole universe (Nifty 500), keeps the
# 47 liquid stocks with the strongest momentum, and uses that shortlist until the
# next refresh. The default 47-stock list is used only until the first refresh works.
#
# Env var (GitHub secret): NTFY_TOPIC
# Optional: MIN_QUICK (default 5), MAX_ALERTS (10), SHORTLIST_SIZE (47),
#           MIN_TURNOVER_CR (10 = Rs 10 crore average daily traded value),
#           MAX_EARLY (3), EARLY_MIN_ACCEL (0.6 = % rise in 15 min), EARLY_MIN_VOL (1.5),
#           COST_PCT (0.15 = estimated round-trip cost in % used by the outcome log),
#           HEARTBEAT (1 = hourly check-in message, 0 = off),
#           MAX_PATTERN (3), FAST_WAIT (120 = seconds until the next scan after an alert),
#           MAX_FAST_STREAK (3 = quick rescans in a row), MAX_EXTRA_REFRESH (6 = extra whole-market
#           rescans a day when the list is quiet), EXTRA_COOLDOWN_MIN (20 = minutes between refreshes)
# Modes: 'python alert_bot.py' = one scan, 'python alert_bot.py summary' = score today's alerts
#        and send the daily summary.

import os
import io
import csv
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
STATE_FILE = "state.json"
WATCHLIST_FILE = "watchlist.json"

MIN_QUICK = float(os.getenv("MIN_QUICK", "5"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "47"))
MIN_TURNOVER_CR = float(os.getenv("MIN_TURNOVER_CR", "10"))
MIN_PRICE = 30
HEARTBEAT = os.getenv("HEARTBEAT", "1") == "1"   # set to 0 to turn off hourly check-ins
MAX_EARLY = int(os.getenv("MAX_EARLY", "3"))                  # max EARLY alerts per scan
EARLY_MIN_ACCEL = float(os.getenv("EARLY_MIN_ACCEL", "0.6"))  # % rise over the last 15 minutes
EARLY_MIN_VOL = float(os.getenv("EARLY_MIN_VOL", "1.5"))      # recent volume vs the normal for this time
COST_PCT = float(os.getenv("COST_PCT", "0.15"))               # estimated round-trip cost (%) for the outcome log
REFRESH_SLOTS = [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30)]   # hourly watchlist refresh
LOG_FILE = "alert_log.csv"
OUTCOME_FILE = "alert_outcomes.csv"
NEXT_WAIT_FILE = "next_wait.txt"                              # read by the workflow loop
MAX_PATTERN = int(os.getenv("MAX_PATTERN", "3"))              # max PATTERN alerts per scan
FAST_WAIT = int(os.getenv("FAST_WAIT", "120"))                # seconds until the next scan after an alert
MAX_FAST_STREAK = int(os.getenv("MAX_FAST_STREAK", "3"))      # quick rescans in a row
MAX_EXTRA_REFRESH = int(os.getenv("MAX_EXTRA_REFRESH", "6"))  # extra whole-market rescans per day
EXTRA_COOLDOWN_MIN = int(os.getenv("EXTRA_COOLDOWN_MIN", "20"))   # minutes between any two refreshes

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


PATTERN_PRIORITY = {"DOUBLE BOTTOM": 0, "ASC TRIANGLE": 1, "BULL FLAG": 2, "BASE BREAKOUT": 3}


def _collapse(idx):
    """Merge swing points that sit within 2 candles of each other (equal highs/lows on a plateau)."""
    out = []
    for i in idx:
        if not out or i - out[-1] > 2:
            out.append(i)
    return out


def find_patterns(t, atr, last):
    """Bullish chart patterns on today's 5-minute candles that have JUST broken out.
    Returns a list of dicts (name, level, low, detail), best pattern first."""
    out = []
    n = len(t)
    if n < 14 or atr <= 0:
        return out
    hi = t["High"].to_numpy(dtype=float)
    lo = t["Low"].to_numpy(dtype=float)
    cl = t["Close"].to_numpy(dtype=float)
    vol = t["Volume"].to_numpy(dtype=float)

    def broke_out(level):
        """Closed above the level within the last two candles and is still close to it."""
        return (last > level + 0.1 * atr and (last - level) <= 3 * atr
                and cl[-4:-1].max() <= level + 0.5 * atr)

    # 1) Base / flag: a tight 8-candle sideways range, then a break up on volume
    B = 8
    base_hi, base_lo = hi[-(B + 2):-2].max(), lo[-(B + 2):-2].min()
    width = base_hi - base_lo
    if width <= 3 * atr and width / last <= 0.01 and broke_out(base_hi):
        base_vol = vol[-(B + 2):-2].mean()
        if base_vol > 0 and vol[-2:].mean() >= 1.3 * base_vol:
            pole = (cl[-(B + 3)] / cl[-(B + 9)] - 1) * 100 if n >= B + 9 else 0.0
            name = "BULL FLAG" if pole >= 1.0 else "BASE BREAKOUT"
            extra = f", pole +{pole:.1f}%" if name == "BULL FLAG" else ""
            out.append({"name": name, "level": base_hi, "low": base_lo,
                        "detail": f"broke above {base_hi:.2f} after a tight base {base_lo:.2f}-{base_hi:.2f}{extra}"})

    # 2) Ascending triangle: flat resistance touched twice or more, rising lows, then a break
    s = max(2, n - 30)
    swing_hi = _collapse([i for i in range(s, n - 2) if hi[i] == hi[i - 2:i + 3].max()])
    swing_lo = _collapse([i for i in range(s, n - 2) if lo[i] == lo[i - 2:i + 3].min()])
    if len(swing_hi) >= 2:
        top = max(hi[i] for i in swing_hi)
        touches = [i for i in swing_hi if hi[i] >= top * (1 - 0.0035)]
        if len(touches) >= 2 and touches[-1] - touches[0] >= 4:
            lows = [i for i in swing_lo if i > touches[0]]
            rising = len(lows) >= 2 and all(lo[b] >= lo[a] + 0.05 * atr for a, b in zip(lows, lows[1:]))
            if rising and broke_out(top):
                out.append({"name": "ASC TRIANGLE", "level": top, "low": lo[lows[-1]],
                            "detail": f"broke above flat resistance {top:.2f} on higher lows "
                                      f"{lo[lows[0]]:.2f} to {lo[lows[-1]]:.2f}"})

    # 3) Double bottom: two lows within 0.3%, at least 6 candles apart, then a break of the middle peak
    s2 = max(2, n - 40)
    lows2 = _collapse([i for i in range(s2, n - 2) if lo[i] == lo[i - 2:i + 3].min()])
    found = False
    for j in range(len(lows2) - 1, 0, -1):
        b = lows2[j]
        if n - b > 25:
            continue
        for k in range(j - 1, -1, -1):
            a = lows2[k]
            if b - a < 6 or abs(lo[a] - lo[b]) / lo[a] > 0.003:
                continue
            neck = hi[a:b + 1].max()
            if neck - max(lo[a], lo[b]) < max(0.006 * last, 2 * atr):
                continue
            if broke_out(neck):
                out.append({"name": "DOUBLE BOTTOM", "level": neck, "low": min(lo[a], lo[b]),
                            "detail": f"broke above the middle peak {neck:.2f} after two lows near {min(lo[a], lo[b]):.2f}"})
                found = True
                break
        if found:
            break

    out.sort(key=lambda p: PATTERN_PRIORITY[p["name"]])
    return out


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

    # Freshness: still within 0.3% of its 30-minute high AND above the previous candle's close
    high30 = float(t["High"].iloc[-6:].max())
    rising = n >= 2 and last > float(t["Close"].iloc[-2])
    fresh = last >= high30 * 0.997 and rising

    # Acceleration: rise over the last 15 minutes, and volume of the last 3 candles vs the
    # same 3 candles on previous days
    accel15 = (last / float(t["Close"].iloc[-4]) - 1) * 100 if n >= 4 else 0.0
    recent3 = float(t["Volume"].iloc[-3:].sum())
    prev3 = [df[df["date"] == d]["Volume"].iloc[max(n - 3, 0):n].sum() for d in days[:-1]]
    avg_prev3 = sum(prev3) / len(prev3) if prev3 else 0
    vol3_x = recent3 / avg_prev3 if avg_prev3 > 0 else 1.0

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
            if fresh:
                entry = tick(last)
                stop, target = tick(last - risk), tick(last + 1.5 * risk)
            else:
                action = "STALE - move is fading, skip"
                why.append("fading from its 30-min high")
    elif score >= 1.5:
        action = "BUILDING - wait for confirmation"

    # EARLY: rising fast on volume, before the slower confirmation signals line up.
    # It tolerates a bigger move away from VWAP than the confirmed alerts do (a fast climb is
    # exactly when the stock is far above VWAP) and ignores RSI (a fast climb always looks
    # overbought). Instead it skips stocks that have already gained a lot: that is chasing.
    early_stretched = ext > 8 or mom30 > 2.5 or from_open > 4
    if (not action.startswith(("BUY NOW", "WATCH")) and not market_closed
            and not too_early and not too_late and not early_stretched
            and n >= 4 and last > vwap and from_open > 0 and fresh
            and accel15 >= EARLY_MIN_ACCEL and vol3_x >= EARLY_MIN_VOL):
        action = "EARLY BUY (rising fast)"
        entry = tick(last)
        stop, target = tick(last - risk), tick(last + 1.5 * risk)
        why.append(f"+{accel15:.1f}% in 15 min on {vol3_x:.1f}x volume")

    # PATTERN: a bullish chart pattern that has just broken out (see find_patterns).
    # The stop sits just under the pattern's low (kept between 1.5 and 4 ATR below), target = 1.5x the risk.
    pattern_name = ""
    patterns = find_patterns(t, atr, last)
    if (patterns and not action.startswith(("BUY NOW", "WATCH")) and not market_closed
            and not too_early and not too_late and not early_stretched
            and last > vwap and rising):
        p = patterns[0]
        pattern_name = p["name"]
        stop = tick(min(max(p["low"] - 0.2 * atr, last - 4 * atr), last - 1.5 * atr))
        target = tick(last + 1.5 * (last - stop))
        entry = tick(last)
        action = f"PATTERN BUY ({p['name']})"
        why.append(f"{p['name']}: {p['detail']}")

    return {"stock": symbol, "price": round(last, 2), "mom30": round(mom30, 2),
            "vol_x": round(vol_ratio, 1), "score": score, "quick": round(quick, 1),
            "rise": rise, "action": action, "entry": entry, "stop": stop,
            "target": target, "why": "; ".join(why),
            "turnover_cr": round(turnover_cr, 1),
            "accel15": round(accel15, 2), "vol3_x": round(vol3_x, 1), "fresh": bool(fresh),
            "pattern": pattern_name,
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
    return {"date": today, "refreshed": [], "stocks": list(DEFAULT_WATCHLIST), "extra": 0, "last_refresh": ""}


def save_watchlist(wl):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f)


def send_heartbeat(results, now):
    """Hourly 'still alive' message with the closest-to-signal stocks."""
    close = sorted([r for r in results if r["score"] > 0],
                   key=lambda r: r["quick"], reverse=True)[:3]
    lines = [f"Scanner running {now.strftime('%H:%M')} IST - no new strong signals.",
             f"Watching {len(results)} stocks."]
    if close:
        lines.append("Closest to a signal:")
        for i, r in enumerate(close, 1):
            lines.append(f"{i}. {r['stock']} {r['price']} (quick {r['quick']}) - {r['action']}")
    lines.append("Next hourly check-in unless a strong signal appears first.")
    notify("\n".join(lines))


# ---------------------------------------------------------------- outcome log

LOG_COLS = ["date", "time", "stock", "kind", "entry", "stop", "target", "quick", "why"]
OUT_COLS = ["date", "time", "stock", "kind", "entry", "stop", "target", "result", "exit",
            "gross_pct", "net_pct", "ret30_pct", "ret60_pct", "mfe_pct", "mae_pct"]


def read_rows(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def append_rows(path, cols, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        for r in rows:
            w.writerow(r)


def log_alert(res, kind, now):
    append_rows(LOG_FILE, LOG_COLS, [[now.date().isoformat(), now.strftime("%H:%M:%S"),
                                      res["stock"], kind, res["entry"], res["stop"],
                                      res["target"], res["quick"], res["why"]]])


def to_ist(df):
    out = df.copy()
    idx = out.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert(IST)
    return out


def simulate(frame, entry_ts, entry, stop, target):
    """Walk the 5-minute candles after an alert (long trade) until 3:15 PM.
    Inside one candle the stop is checked before the target (the cautious choice).
    Returns None when there is no price data after the alert."""
    first = entry_ts.replace(second=0, microsecond=0)
    first = first - timedelta(minutes=first.minute % 5) + timedelta(minutes=5)
    cutoff = entry_ts.replace(hour=15, minute=15, second=0, microsecond=0)
    c = frame[(frame.index >= first) & (frame.index < cutoff)]
    if c.empty:
        return None
    result, exit_px, exit_i = "OPEN AT 3:15", float(c["Close"].iloc[-1]), len(c) - 1
    for i in range(len(c)):
        lo, hi = float(c["Low"].iloc[i]), float(c["High"].iloc[i])
        if lo <= stop:
            result, exit_px, exit_i = "STOP", stop, i
            break
        if hi >= target:
            result, exit_px, exit_i = "TARGET", target, i
            break
    seg = c.iloc[:exit_i + 1]

    def ret_at(minutes):
        row = c[c.index == first + timedelta(minutes=minutes - 5)]
        return (float(row["Close"].iloc[0]) / entry - 1) * 100 if not row.empty else None

    gross = (exit_px / entry - 1) * 100
    return {"result": result, "exit": exit_px, "gross": gross, "net": gross - COST_PCT,
            "ret30": ret_at(30), "ret60": ret_at(60),
            "mfe": (float(seg["High"].max()) / entry - 1) * 100,
            "mae": (float(seg["Low"].min()) / entry - 1) * 100}


def evaluate_day(day):
    """Score the day's alerts that have no outcome yet. Returns how many were added."""
    done = {(r["date"], r["time"], r["stock"], r["kind"]) for r in read_rows(OUTCOME_FILE)}
    todo = [r for r in read_rows(LOG_FILE)
            if r["date"] == day and (r["date"], r["time"], r["stock"], r["kind"]) not in done]
    if not todo:
        return 0
    frames = {}
    syms = sorted({r["stock"] for r in todo})
    for i in range(0, len(syms), 50):
        frames.update(fetch_batch(syms[i:i + 50]))

    def cell(v):
        return "" if v is None else round(v, 2)

    out = []
    for r in todo:
        d = frames.get(r["stock"])
        if d is None:
            print("no price data for", r["stock"])
            continue
        try:
            frame = to_ist(d)[["Open", "High", "Low", "Close"]].dropna()
            ts = datetime.strptime(f"{r['date']} {r['time']}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            sim = simulate(frame, ts, float(r["entry"]), float(r["stop"]), float(r["target"]))
        except Exception as e:
            print("could not score", r["stock"], e)
            continue
        if not sim:
            print("no candles after the alert for", r["stock"])
            continue
        out.append([r["date"], r["time"], r["stock"], r["kind"], r["entry"], r["stop"], r["target"],
                    sim["result"], cell(sim["exit"]), cell(sim["gross"]), cell(sim["net"]),
                    cell(sim["ret30"]), cell(sim["ret60"]), cell(sim["mfe"]), cell(sim["mae"])])
    if out:
        append_rows(OUTCOME_FILE, OUT_COLS, out)
    return len(out)


def _stats(rows):
    nets = [float(r["net_pct"]) for r in rows]
    n = len(nets)
    return {"n": n, "win": 100.0 * sum(1 for x in nets if x > 0) / n,
            "avg": sum(nets) / n, "sum": sum(nets)}


def run_summary():
    """Score today's alerts and send the daily summary (also used for a manual re-run)."""
    now = datetime.now(IST)
    day = os.getenv("SUMMARY_DATE") or now.date().isoformat()
    evaluate_day(day)
    outcomes = read_rows(OUTCOME_FILE)
    today_rows = [r for r in outcomes if r["date"] == day]
    lines = [f"NSE alert summary {day} (simulated through 3:15 PM)"]
    if today_rows:
        count = {k: sum(1 for r in today_rows if r["result"] == k)
                 for k in ("TARGET", "STOP", "OPEN AT 3:15")}
        s = _stats(today_rows)
        lines.append(f"Alerts: {s['n']} -> {count['TARGET']} hit target, {count['STOP']} hit stop, "
                     f"{count['OPEN AT 3:15']} still open at 3:15")
        lines.append(f"After est. {COST_PCT:.2f}% cost per trade: average {s['avg']:+.2f}% per alert, "
                     f"total {s['sum']:+.2f}%")
        for kind in ("CONFIRMED", "WATCH", "EARLY", "PATTERN"):
            rows = [r for r in today_rows if r["kind"] == kind]
            if rows:
                k = _stats(rows)
                lines.append(f"  {kind}: {k['n']} alerts, {k['win']:.0f}% positive, average {k['avg']:+.2f}%")
        best = max(today_rows, key=lambda r: float(r["net_pct"]))
        worst = min(today_rows, key=lambda r: float(r["net_pct"]))
        lines.append(f"Best: {best['stock']} {float(best['net_pct']):+.2f}% ({best['result']}) | "
                     f"Worst: {worst['stock']} {float(worst['net_pct']):+.2f}% ({worst['result']})")
    elif any(r["date"] == day for r in read_rows(LOG_FILE)):
        lines.append("Alerts were sent, but no price data was available to score them.")
    else:
        lines.append("No alerts today.")
    if outcomes:
        a = _stats(outcomes)
        days = len({r["date"] for r in outcomes})
        lines.append(f"All days so far ({days} days, {a['n']} alerts): {a['win']:.0f}% positive, "
                     f"average {a['avg']:+.2f}% per alert, total {a['sum']:+.2f}%")
        for kind in ("CONFIRMED", "WATCH", "EARLY", "PATTERN"):
            rows = [r for r in outcomes if r["kind"] == kind]
            if rows:
                k = _stats(rows)
                lines.append(f"  {kind}: {k['n']} alerts, {k['win']:.0f}% positive, average {k['avg']:+.2f}%")
    lines.append("Simulated: assumes fills at the alert price and stop checked first. Real results will differ.")
    notify("\n".join(lines))


def is_actionable(res):
    """Would this stock produce an alert right now (ignoring 'already alerted today')?"""
    act = res["action"]
    if act.startswith(("BUY NOW", "WATCH")):
        return res["quick"] >= MIN_QUICK
    return act.startswith(("EARLY BUY", "PATTERN BUY"))


def list_is_quiet(results):
    """True when we have data but nothing in the current list is actionable."""
    return bool(results) and not any(is_actionable(r) for r in results)


def extra_refresh_allowed(wl, now):
    """Extra whole-market rescans: only 9:50-14:30, limited per day, and not too soon after a refresh."""
    if not ((9, 50) <= (now.hour, now.minute) < (14, 30)):
        return False
    if wl.get("extra", 0) >= MAX_EXTRA_REFRESH:
        return False
    last = wl.get("last_refresh")
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() / 60 < EXTRA_COOLDOWN_MIN:
                return False
        except Exception:
            pass
    return True


def refresh_watchlist(slot, wl, now, force):
    """Scan the whole universe and keep the strongest stocks. Returns (results, announced)."""
    wl["last_refresh"] = now.isoformat()
    silent = slot.startswith("X")          # extra refreshes triggered by a quiet list stay silent
    announced = False
    universe = get_universe()
    all_results = analyse_many(universe)
    picks = pick_shortlist(all_results)
    wl["refreshed"].append(slot)
    if len(picks) >= 10:
        wl["stocks"] = [r["stock"] for r in picks]
        results = picks
        names = ", ".join(r["stock"] for r in picks[:15])
        print(f"Watchlist refreshed {now.strftime('%H:%M')}: {len(picks)} stocks. Top: {names}")
        if (len(wl["refreshed"]) == 1 or force) and not silent:   # only announce the first one of the day
            notify(f"Watchlist refreshed {now.strftime('%H:%M')} IST. "
                   f"Scanned {len(all_results)} of {len(universe)} stocks, "
                   f"shortlisted {len(picks)} with the strongest momentum. "
                   f"It will refresh hourly until 1:30 PM.\n"
                   f"Top: {names}")
            announced = True
    else:
        print("Shortlist too small; keeping the current watchlist.")
        if not silent:
            notify(f"Watchlist refresh at {now.strftime('%H:%M')} IST failed "
                   f"(only {len(all_results)} stocks had data). Keeping the previous list.")
            announced = True
        results = analyse_many(wl["stocks"])
    save_watchlist(wl)
    return results, announced


def set_next_wait(state, today, alerted):
    """Tell the workflow loop how long to wait before the next scan.
    After an alert: FAST_WAIT seconds, up to MAX_FAST_STREAK scans in a row. Otherwise the normal 5 minutes
    (no file)."""
    streak = 0
    try:
        prev = str(state.get("_fast", ""))
        if prev.startswith(today):
            streak = int(prev.split("|")[1])
    except Exception:
        streak = 0
    if alerted:
        streak += 1
        state["_fast"] = f"{today}|{streak}"
        if streak <= MAX_FAST_STREAK:
            with open(NEXT_WAIT_FILE, "w") as f:
                f.write(str(FAST_WAIT))
            return
    else:
        state["_fast"] = f"{today}|0"
    if os.path.exists(NEXT_WAIT_FILE):
        os.remove(NEXT_WAIT_FILE)


# ---------------------------------------------------------------- main

def main():
    if (len(sys.argv) > 1 and sys.argv[1] == "summary") or os.getenv("SUMMARY", "").lower() == "true":
        run_summary()
        return

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

    if os.path.exists(NEXT_WAIT_FILE):
        os.remove(NEXT_WAIT_FILE)   # stale hint from the previous scan

    today = now.date().isoformat()
    state = load_state(today)
    wl = load_watchlist(today)

    # Decide whether the watchlist should be refreshed on this run (hourly, 9:30 to 1:30)
    slot = None
    if force_refresh:
        slot = "M"
    else:
        for i, (h, m) in enumerate(REFRESH_SLOTS):
            nxt = REFRESH_SLOTS[i + 1] if i + 1 < len(REFRESH_SLOTS) else (14, 30)
            sid = f"{h:02d}{m:02d}"
            if (h, m) <= hm < nxt and sid not in wl["refreshed"]:
                slot = sid
                break

    sent_note = False
    if slot:
        results, sent_note = refresh_watchlist(slot, wl, now, force_refresh)
    else:
        results = analyse_many(wl["stocks"])
        # Nothing worth alerting in the current list -> look for the movers in the whole market again
        if list_is_quiet(results) and extra_refresh_allowed(wl, now):
            print("No strong movement in the current list - rescanning the whole market.")
            wl["extra"] = wl.get("extra", 0) + 1
            slot = "X" + now.strftime("%H%M")
            results, sent_note = refresh_watchlist(slot, wl, now, False)

    hour_tag = f"{today}|{now.hour:02d}"
    if sent_note:
        state["_heartbeat"] = hour_tag   # that note already counts as a check-in

    candidates, early, patt = [], [], []
    for res in results:
        act = res["action"]
        # CONFIRMED / WATCH: actionable buys with strong quick-rise strength
        if act.startswith(("BUY NOW", "WATCH")) and res["quick"] >= MIN_QUICK:
            key = f"{res['stock']}:BUY"
            prev = state.get(key)
            # New today, or upgraded from STRONG to HOT
            if prev is None or (str(prev).endswith("STRONG") and res["rise"] == "HOT"):
                kind = "CONFIRMED" if act.startswith("BUY NOW") else "WATCH"
                candidates.append((key, res, kind))
        # PATTERN: chart pattern breakout, once per stock per day
        elif act.startswith("PATTERN BUY"):
            key = f"{res['stock']}:PATTERN"
            if state.get(key) is None:
                patt.append((key, res, "PATTERN"))
        # EARLY: rising fast on volume, once per stock per day
        elif act.startswith("EARLY BUY"):
            key = f"{res['stock']}:EARLY"
            if state.get(key) is None:
                early.append((key, res, "EARLY"))

    candidates.sort(key=lambda x: x[1]["quick"], reverse=True)
    early.sort(key=lambda x: x[1]["accel15"], reverse=True)
    patt.sort(key=lambda x: x[1]["quick"], reverse=True)
    candidates = (candidates + patt[:MAX_PATTERN] + early[:MAX_EARLY])[:MAX_ALERTS]

    if not candidates:
        print("No new buy signals.", now.strftime("%H:%M"), "stocks scanned:", len(results))
        if HEARTBEAT and state.get("_heartbeat") != hour_tag:
            send_heartbeat(results, now)
            state["_heartbeat"] = hour_tag
        set_next_wait(state, today, False)
        save_state(state)
        return

    lines = [f"NSE BUY alerts {now.strftime('%H:%M')} IST", ""]
    for i, (key, res, kind) in enumerate(candidates, 1):
        tag = f"  - {res['pattern']}" if kind == "PATTERN" and res.get("pattern") else ""
        lines.append(f"{i}. [{kind}] {res['stock']} @ {res['entry']}  (quick {res['quick']}){tag}")
        lines.append(f"   Stop {res['stop']} | Target {res['target']}")
        lines.append(f"   30-min {res['mom30']:+.1f}% | 15-min {res['accel15']:+.1f}% | volume {res['vol_x']}x")
        lines.append(f"   {res['why']}")
        lines.append("")
    lines.append("CONFIRMED = breakout + volume + trend. WATCH = leaning up, not confirmed.")
    lines.append("EARLY = rising fast on volume, not yet confirmed (riskier).")
    lines.append("PATTERN = chart pattern breakout (flag, base, triangle, double bottom); stop sits under the pattern.")
    lines.append("Momentum signals, not a guaranteed rise. Not financial advice. Always use the stop-loss.")

    alerted = False
    if notify("\n".join(lines)):
        alerted = True
        for key, res, kind in candidates:
            state[key] = f"{today}|{kind}" if kind in ("EARLY", "PATTERN") else f"{today}|{res['rise']}"
            log_alert(res, kind, now)
        state["_heartbeat"] = hour_tag   # an alert counts as a check-in too
    set_next_wait(state, today, alerted)
    save_state(state)


if __name__ == "__main__":
    main()
