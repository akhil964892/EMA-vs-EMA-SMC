# EdgeScan — Long & Short, Fixed Lot, Prev-Bar SL • EMA vs EMA+SMC
# - Data: Yahoo Finance (yfinance), crypto fallback via Binance klines
# - ML: GradientBoosting (calibrated), walk-forward; two models trained separately
# - Signals:
#     Long  if proba >= thr
#     Short if proba <= 1 - thr  (toggle with checkbox)
# - Filters (togglable):
#     EMA model: long if EMA9>EMA15, short if EMA9<EMA15
#     SMC model: long if bullish SMC context, short if bearish SMC context
# - SL/TP:
#     Long : SL = prev Low;  TP = entry + rr*(entry - prev_low)
#     Short: SL = prev High; TP = entry - rr*(prev_high - entry)
#   rr in {1,2,3}; fixed lot (no compounding)
# - Outputs (per model): metrics, equity curve, PnL-by-RR (by side), trade log, per-year by side
# - Forecast panel (per model): Long & Short plans with entry/SL/TP1/2/3

import math, warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="EdgeScan — Long & Short (Prev-Bar SL)", layout="wide")
st.title("EdgeScan — Long & Short (Prev-Bar SL) • EMA vs EMA+SMC")
st.caption("Fixed lot • Prev Low/High as SL • RR 1/2/3 • Walk-forward ML • Separate models with separate results")

# -- Utils --
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.rolling(n, min_periods=1).mean()

def safe_z(x: pd.Series) -> pd.Series:
    s = (x - x.mean()) / (x.std() + 1e-9)
    return s.replace([np.inf, -np.inf], 0).fillna(0)

def drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    rm = equity.cummax()
    dd = equity / rm - 1.0
    return float(dd.min()), dd

def sharpe(returns: pd.Series, ppy: int = 252) -> float:
    r = returns.fillna(0)
    mu = r.mean() * ppy
    sd = (r.std() + 1e-12) * math.sqrt(ppy)
    return float(mu / sd)

def cagr(equity: pd.Series, ppy: int = 252) -> float:
    if len(equity) < 2: return 0.0
    yrs = len(equity) / ppy
    if yrs <= 0: return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1/yrs) - 1)

def profit_factor(rets: pd.Series) -> float:
    g = rets[rets > 0].sum()
    l = -rets[rets < 0].sum()
    return float(g / (l + 1e-12))

# ----------------- Data -----------------
def looks_like_crypto(sym: str) -> bool:
    s = sym.upper().strip()
    return s.endswith("-USD") or s in {
        "BTC-USD","ETH-USD","SOL-USD","ADA-USD","XRP-USD","DOGE-USD",
        "BNB-USD","DOT-USD","LTC-USD","AVAX-USD","MATIC-USD"
    }

def binance_pair(sym: str) -> str:
    # "BTC-USD" -> "BTCUSDT"
    s = sym.upper().replace("-", "")
    base = s[:-3] if s.endswith("USD") else s
    return f"{base}USDT"

@st.cache_data(show_spinner=False)
def fetch_binance_klines(sym: str, limit: int = 1200) -> pd.DataFrame:
    pair = binance_pair(sym)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": pair, "interval": "1d", "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for k in data:
        rows.append({
            "Date": pd.to_datetime(k[0], unit="ms"),
            "Open": float(k[1]), "High": float(k[2]), "Low": float(k[3]),
            "Close": float(k[4]), "Volume": float(k[5]),
        })
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    out = df[cols].copy().reset_index()
    out.rename(columns={out.columns[0]: "Date"}, inplace=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out.dropna(subset=["Open","High","Low","Close"])

@st.cache_data(show_spinner=False)
def fetch_ohlcv(sym: str, start: str, end: str) -> pd.DataFrame:
    try:
        d0 = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False, threads=False, interval="1d")
        df = _normalize_yf(d0)
        if not df.empty: return df
    except Exception:
        pass
    try:
        tk = yf.Ticker(sym)
        h = tk.history(period="max", interval="1d", auto_adjust=False, actions=False)
        df = _normalize_yf(h)
        if not df.empty:
            df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))].reset_index(drop=True)
            if not df.empty: return df
    except Exception:
        pass
    if looks_like_crypto(sym):
        return fetch_binance_klines(sym)
    raise RuntimeError("No data from Yahoo (and not crypto). Try a different symbol or wider dates.")

# -- SMC-lite features --
def detect_swings(df: pd.DataFrame, lb=3, lf=3) -> pd.DataFrame:
    df = df.copy(); n = len(df)
    sh = np.zeros(n, dtype=int); sl = np.zeros(n, dtype=int)
    for i in range(lb, n-lf):
        w = df.iloc[i-lb:i+lf+1]
        if df["High"].iloc[i] == w["High"].max(): sh[i] = 1
        if df["Low" ].iloc[i] == w["Low" ].min(): sl[i] = 1
    df["swing_high"] = sh; df["swing_low"] = sl; return df

def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); last_h = np.nan; last_l = np.nan
    bos_up = np.zeros(len(df), dtype=int); bos_dn = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if df["swing_high"].iat[i] == 1: last_h = df["High"].iat[i]
        if df["swing_low" ].iat[i] == 1: last_l = df["Low"].iat[i]
        if not np.isnan(last_h) and df["Close"].iat[i] > last_h: bos_up[i] = 1
        if not np.isnan(last_l) and df["Close"].iat[i] < last_l: bos_dn[i] = 1
    df["bos_up"] = bos_up; df["bos_dn"] = bos_dn
    tr_up = pd.Series(bos_up).rolling(10).sum()
    tr_dn = pd.Series(bos_dn).rolling(10).sum()
    df["choch"] = np.where(tr_up > tr_dn, 1, np.where(tr_dn > tr_up, -1, 0))
    return df

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); up = np.zeros(len(df), dtype=int); dn = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        if df["Low" ].iat[i] > df["High"].iat[i-2]: up[i] = 1
        if df["High"].iat[i] < df["Low" ].iat[i-2]: dn[i] = 1
    df["fvg_up"] = up; df["fvg_dn"] = dn; return df

def detect_order_blocks(df: pd.DataFrame, lookback=50) -> pd.DataFrame:
    df = df.copy()
    bull_ob = np.full(len(df), np.nan); bear_ob = np.full(len(df), np.nan)
    last_down = None; last_up = None
    for i in range(1, len(df)):
        if df["Close"].iat[i] < df["Open"].iat[i]: last_down = i
        elif df["Close"].iat[i] > df["Open"].iat[i]: last_up = i
        if df["bos_up"].iat[i] == 1 and last_down is not None and i-last_down<=lookback:
            bull_ob[i] = df["Open"].iat[last_down]
        if df["bos_dn"].iat[i] == 1 and last_up is not None and i-last_up<=lookback:
            bear_ob[i] = df["Open"].iat[last_up]
    df["bull_ob"] = bull_ob; df["bear_ob"] = bear_ob
    df["dist_bull_ob"] = (df["Close"] - df["bull_ob"]).abs()
    df["dist_bear_ob"] = (df["Close"] - df["bear_ob"]).abs()
    df["near_bull_ob"] = (df["dist_bull_ob"] / (df["Close"] + 1e-9) < 0.005).astype(int)
    df["near_bear_ob"] = (df["dist_bear_ob"] / (df["Close"] + 1e-9) < 0.005).astype(int)
    return df

# -- Features --
def build_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA9"]  = ema(out["Close"], 9);  out["EMA15"] = ema(out["Close"], 15)
    out["ATR14"] = atr(out, 14)
    out["ema_spread"] = (out["EMA9"] - out["EMA15"]) / (out["Close"] + 1e-9)
    out["ema_slope"]  = out["EMA9"].pct_change(3)
    out["ret_1"] = out["Close"].pct_change(1); out["ret_5"] = out["Close"].pct_change(5)
    out["vol_z"] = safe_z(out["Volume"].fillna(0))
    return out

def build_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_ema_features(df)
    out = detect_swings(out, 3, 3); out = detect_structure(out); out = detect_fvg(out); out = detect_order_blocks(out)
    out["mit_up"] = ((out["fvg_up"]==1)&(out["near_bull_ob"]==1)&(out["bos_up"]==1)).astype(int)
    out["mit_dn"] = ((out["fvg_dn"]==1)&(out["near_bear_ob"]==1)&(out["bos_dn"]==1)).astype(int)
    return out

# -- Labels --
def make_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    return (fwd_ret > 0.0).astype(int)

# -- Model --
def fit_clf(X: pd.DataFrame, y: pd.Series, rnd=7):
    base = GradientBoostingClassifier(
        n_estimators=220, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=rnd
    )
    if y.nunique() < 2:
        base.fit(X, y)
        class W:
            def __init__(self, m): self.m=m
            def predict_proba(self, X): return self.m.predict_proba(X)
        return W(base)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    return clf

def walk_forward_probs(X: pd.DataFrame, y: pd.Series, min_train=400, n_folds=4) -> np.ndarray:
    n = len(X)
    probs = np.full(n, np.nan)
    fold_size = max(1, (n - min_train) // max(n_folds, 1))
    folds = []
    for k in range(n_folds):
        tr_end = min_train + fold_size * k
        te_end = min_train + fold_size * (k+1)
        tr = np.arange(0, max(tr_end, 1))
        te = np.arange(max(tr_end, 1), min(te_end, n))
        if len(te) > 0:
            folds.append((tr, te))
    if not folds:
        half = max(1, n//2)
        folds = [(np.arange(0, half), np.arange(half, n))]
    for (tr, te) in folds:
        clf = fit_clf(X.iloc[tr], y.iloc[tr])
        probs[te] = clf.predict_proba(X.iloc[te])[:,1]
    return probs

def metrics(y_true, y_proba, thr=0.6) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_proba = pd.Series(y_proba).clip(0,1).fillna(0.5).values
    y_pred = (y_proba >= thr).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true))==2 else float("nan"),
        "MCC": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true))==2 else float("nan"),
    }
    return {k: float(v) for k,v in out.items()}

# -- Simulator (Prev Low/High SL, Fixed Lot, Long & Short) --
def simulate_prev_bar_fixedlot_long_short(
    df: pd.DataFrame,
    proba: pd.Series,
    thr: float = 0.6,
    horizon: int = 5,
    lot: float = 10.0,
    rr_list=(1,2,3),
    long_mask: pd.Series | None = None,
    short_mask: pd.Series | None = None,
    initial_capital: float = 100000.0,
    enable_shorts: bool = True
) -> Dict[str, Any]:
    close, openp, high, low = df["Close"].values, df["Open"].values, df["High"].values, df["Low"].values
    dates = pd.to_datetime(df["Date"]).values

    long_sig  = (proba.values >= thr).astype(int)
    short_sig = (proba.values <= (1.0 - thr)).astype(int) if enable_shorts else np.zeros_like(long_sig)

    if long_mask  is not None:  long_sig  = long_sig  * long_mask.astype(int).values
    if short_mask is not None:  short_sig = short_sig * short_mask.astype(int).values

    eq = float(initial_capital)
    trades = []
    MAX_PNL = 5.0 * initial_capital 

    for i in range(len(df) - horizon - 1):
        side = None

        # ---- LONG ----
        if long_sig[i] == 1:
            side = "LONG"
            entry_idx = i + 1
            entry = float(openp[entry_idx])
            prev_bar = float(low[i])
            risk = entry - prev_bar
            if not (np.isfinite(entry) and np.isfinite(prev_bar)) or risk <= 1e-9:
                continue
            sl = prev_bar
            tps = {rr: entry + rr * risk for rr in rr_list}

            exit_price = None; exit_idx = None; tag = None; rr_hit = 0
            for j in range(1, horizon+1):
                b = i + j; lo, hi = float(low[b]), float(high[b])
                if lo <= sl:
                    exit_price, exit_idx, tag, rr_hit = sl, b, "SL", 0
                    break
                hit = next((rr for rr in sorted(rr_list) if hi >= tps[rr]), None)
                if hit is not None:
                    exit_price, exit_idx, tag, rr_hit = tps[hit], b, f"TP{hit}", hit
                    break
            if exit_price is None:
                exit_idx = i + horizon; exit_price = float(close[exit_idx]); tag = "TIME"; rr_hit = 0
            pnl_money = float(lot * (exit_price - entry))

        # ---- SHORT ----
        elif enable_shorts and short_sig[i] == 1:
            side = "SHORT"
            entry_idx = i + 1
            entry = float(openp[entry_idx])
            prev_bar = float(high[i])
            risk = prev_bar - entry
            if not (np.isfinite(entry) and np.isfinite(prev_bar)) or risk <= 1e-9:
                continue
            sl = prev_bar
            tps = {rr: entry - rr * risk for rr in rr_list}

            exit_price = None; exit_idx = None; tag = None; rr_hit = 0
            for j in range(1, horizon+1):
                b = i + j; lo, hi = float(low[b]), float(high[b])
                if hi >= sl:
                    exit_price, exit_idx, tag, rr_hit = sl, b, "SL", 0
                    break
                hit = next((rr for rr in sorted(rr_list) if lo <= tps[rr]), None)
                if hit is not None:
                    exit_price, exit_idx, tag, rr_hit = tps[hit], b, f"TP{hit}", hit
                    break
            if exit_price is None:
                exit_idx = i + horizon; exit_price = float(close[exit_idx]); tag = "TIME"; rr_hit = 0
            pnl_money = float(lot * (entry - exit_price))  # short pnl

        else:
            continue

        pnl_money = float(np.clip(pnl_money, -MAX_PNL, MAX_PNL))
        eq = float(eq + pnl_money)

        trades.append({
            "side": side,
            "entry_date": pd.to_datetime(dates[entry_idx]),
            "exit_date":  pd.to_datetime(dates[exit_idx]),
            "rr_used": int(rr_hit),
            "lot": float(lot),
            "entry": round(entry, 5),
            "sl_prev_bar": round(sl, 5),
            "exit": round(float(exit_price), 5),
            "pnl": round(pnl_money, 2),
            "equity_after": round(eq, 2),
            "win": 1 if pnl_money > 0 else 0
        })

    log = pd.DataFrame(trades)
    equity_curve = pd.Series([initial_capital] + [t["equity_after"] for t in trades], name="Equity")

    if log.empty:
        return {"log": log, "equity": equity_curve, "summary": {}, "per_year": pd.DataFrame(), "rr_perf": pd.DataFrame()}

    # ---- per-year by side ----
    log["year"] = pd.to_datetime(log["exit_date"]).dt.year
    per_year = log.groupby(["year","side"], as_index=False).agg(
        trades=("pnl","count"),
        wins=("win","sum"),
        pnl=("pnl","sum")
    )
    per_year["winrate%"] = (per_year["wins"] / per_year["trades"].clip(lower=1)) * 100.0
    per_year["pnl_pct_of_capital"] = (per_year["pnl"] / float(initial_capital)) * 100.0

    # ---- rr performance by side ----
    rr_perf = log.groupby(["side","rr_used"], as_index=False).agg(
        trades=("pnl","count"), pnl=("pnl","sum"), wins=("win","sum")
    )
    rr_perf["winrate%"] = (rr_perf["wins"] / rr_perf["trades"].clip(lower=1)) * 100.0

    # ---- summary ----
    dd_min, _ = drawdown(equity_curve)
    trade_rets = log["pnl"] / float(initial_capital)
    summary = {
        "Trades": int(len(log)),
        "Wins": int(log["win"].sum()),
        "Losses": int((1-log["win"]).sum()),
        "WinRate%": round(100*log["win"].mean(), 2),
        "FinalEquity": round(float(equity_curve.iloc[-1]), 2),
        "TotalPnL": round(float(equity_curve.iloc[-1] - initial_capital), 2),
        "MaxDD%": round(100*dd_min, 2),
        "CAGR%": round(100*cagr(equity_curve), 2),
        "Sharpe": round(sharpe(trade_rets.fillna(0)), 2),
        "ProfitFactor": round(profit_factor(trade_rets.fillna(0)), 2),
    }

    return {"log": log, "equity": equity_curve, "summary": summary, "per_year": per_year, "rr_perf": rr_perf}

# ----------------- Pipeline (per model) -----------------
def run_model_long_short(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
    thr: float, horizon: int, lot: float, init_cap: float,
    allow_shorts: bool, use_filters: bool,
    long_mask: pd.Series | None, short_mask: pd.Series | None
):
    proba = pd.Series(walk_forward_probs(X, y, min_train=min(600, max(400, len(df)//2)), n_folds=4)).fillna(0.5)

    lm = long_mask if use_filters else None
    sm = short_mask if use_filters else None

    bt = simulate_prev_bar_fixedlot_long_short(
        df=df, proba=proba, thr=thr, horizon=horizon, lot=lot,
        rr_list=(1,2,3), long_mask=lm, short_mask=sm,
        initial_capital=init_cap, enable_shorts=allow_shorts
    )
    mets = metrics(y, proba, thr=thr)
    return proba, mets, bt

# ----------------- Forecast helpers -----------------
def latest_plan_long(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty: return {}
    c = float(df["Close"].iloc[-1]); lo = float(df["Low"].iloc[-1])
    risk = max(c - lo, 1e-9)
    return {"entry": round(c,5), "sl": round(lo,5), "tp1": round(c + 1*risk,5),
            "tp2": round(c + 2*risk,5), "tp3": round(c + 3*risk,5), "risk_dist": round(risk,5)}

def latest_plan_short(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty: return {}
    c = float(df["Close"].iloc[-1]); hi = float(df["High"].iloc[-1])
    risk = max(hi - c, 1e-9)
    return {"entry": round(c,5), "sl": round(hi,5), "tp1": round(c - 1*risk,5),
            "tp2": round(c - 2*risk,5), "tp3": round(c - 3*risk,5), "risk_dist": round(risk,5)}

# ----------------- UI -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.text_input("Symbol", value="AAPL")   # e.g., BTC-USD, RELIANCE.NS, ^NSEI
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", pd.to_datetime("2016-01-01")).strftime("%Y-%m-%d")
    with c2:
        end = st.date_input("End", pd.Timestamp.today()).strftime("%Y-%m-%d")
    horizon = st.slider("Holding Horizon (days)", 3, 20, 5, 1)
    lot = st.number_input("Fixed Lot (units/shares)", min_value=0.1, max_value=1_000_000.0, value=10.0, step=1.0)
    thr = st.slider("Probability Threshold (↑ lowers DD, ↓ more trades)", 0.50, 0.80, 0.60, 0.01)
    allow_shorts = st.checkbox("Enable Shorts (sell)", value=True)
    use_filters  = st.checkbox("Use Trend/Context Filters (lower DD)", value=True)
    init_cap = st.number_input("Initial Capital", min_value=10_000, max_value=10_000_000, value=100_000, step=10_000)
    run = st.button("Run", type="primary", use_container_width=True)

tabs = st.tabs(["EMA Model", "EMA+SMC Model"])

if run:
    try:
        # ---- load ----
        df = fetch_ohlcv(symbol, start, end)
        if len(df) < 300:
            raise RuntimeError("Dataset too small (<300 bars). Choose a longer period.")
        df["ATR14"] = atr(df, 14)

        # ---- features ----
        fe_ema = build_ema_features(df)
        fe_smc = build_smc_features(df)
        y = make_labels(df, horizon=horizon)

        # ---- EMA matrices & masks ----
        cols_ema = ["EMA9","EMA15","ATR14","ema_spread","ema_slope","ret_1","ret_5","vol_z"]
        X_ema_all = fe_ema[cols_ema].copy()
        m_ema = X_ema_all.notna().all(1) & y.notna()
        df_ema = df.loc[m_ema].reset_index(drop=True)
        X_ema = X_ema_all.loc[m_ema].reset_index(drop=True)
        y_ema = y.loc[m_ema].reset_index(drop=True)
        trend_long_ema  = (fe_ema.loc[m_ema, "EMA9"] > fe_ema.loc[m_ema, "EMA15"]).reset_index(drop=True)
        trend_short_ema = (fe_ema.loc[m_ema, "EMA9"] < fe_ema.loc[m_ema, "EMA15"]).reset_index(drop=True)

        # ---- SMC matrices & masks ----
        cols_smc = cols_ema + ["swing_high","swing_low","bos_up","bos_dn","choch","fvg_up","fvg_dn",
                               "bull_ob","bear_ob","dist_bull_ob","dist_bear_ob","near_bull_ob","near_bear_ob","mit_up","mit_dn"]
        X_smc_all = fe_smc[cols_smc].copy()
        for c in ["bull_ob","bear_ob","dist_bull_ob","dist_bear_ob"]:
            if c in X_smc_all.columns:
                X_smc_all[c] = X_smc_all[c].fillna(X_smc_all[c].median())
        m_smc = X_smc_all.notna().all(1) & y.notna()
        df_smc = df.loc[m_smc].reset_index(drop=True)
        X_smc = X_smc_all.loc[m_smc].reset_index(drop=True)
        y_smc = y.loc[m_smc].reset_index(drop=True)

        # bullish context (long) and bearish context (short)
        ctx_long_smc = (
            ((fe_smc.loc[m_smc, "bos_up"] == 1) | (fe_smc.loc[m_smc, "mit_up"] == 1) |
             (fe_smc.loc[m_smc, "near_bull_ob"] == 1) | (fe_smc.loc[m_smc, "fvg_up"] == 1)) &
            (fe_smc.loc[m_smc, "bos_dn"] == 0)
        ).reset_index(drop=True)
        ctx_short_smc = (
            ((fe_smc.loc[m_smc, "bos_dn"] == 1) | (fe_smc.loc[m_smc, "mit_dn"] == 1) |
             (fe_smc.loc[m_smc, "near_bear_ob"] == 1) | (fe_smc.loc[m_smc, "fvg_dn"] == 1)) &
            (fe_smc.loc[m_smc, "bos_up"] == 0)
        ).reset_index(drop=True)

        # ---- Run EMA Model ----
        proba_ema, mets_ema, bt_ema = run_model_long_short(
            df=df_ema, X=X_ema, y=y_ema, thr=thr, horizon=horizon, lot=float(lot), init_cap=float(init_cap),
            allow_shorts=allow_shorts, use_filters=use_filters,
            long_mask=trend_long_ema, short_mask=trend_short_ema
        )

        # ---- Run EMA+SMC Model ----
        proba_smc, mets_smc, bt_smc = run_model_long_short(
            df=df_smc, X=X_smc, y=y_smc, thr=thr, horizon=horizon, lot=float(lot), init_cap=float(init_cap),
            allow_shorts=allow_shorts, use_filters=use_filters,
            long_mask=ctx_long_smc, short_mask=ctx_short_smc
        )

        # --------------- EMA TAB ----------------
        with tabs[0]:
            st.subheader("Metrics — EMA")
            st.json({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in mets_ema.items()})

            st.subheader("Forecast Plan — EMA")
            planL = latest_plan_long(df_ema); planS = latest_plan_short(df_ema)
            c = st.columns(5)
            if planL:
                c[0].metric("Long Entry", planL["entry"]); c[1].metric("Long SL", planL["sl"])
                c[2].metric("TP1", planL["tp1"]); c[3].metric("TP2", planL["tp2"]); c[4].metric("TP3", planL["tp3"])
                st.caption(f"Long risk distance: {planL['risk_dist']}")
            if planS:
                c2 = st.columns(5)
                c2[0].metric("Short Entry", planS["entry"]); c2[1].metric("Short SL", planS["sl"])
                c2[2].metric("TP1", planS["tp1"]); c2[3].metric("TP2", planS["tp2"]); c2[4].metric("TP3", planS["tp3"])
                st.caption(f"Short risk distance: {planS['risk_dist']}")

            st.subheader("Backtest Charts — EMA")
            rr = bt_ema["rr_perf"]
            if rr.empty:
                st.warning("EMA: No trades. Try wider dates, lower threshold, or different symbol.")
            else:
                # PnL by RR & side
                try:
                    piv = rr.pivot(index="rr_used", columns="side", values="pnl").fillna(0.0)
                except Exception:
                    piv = pd.DataFrame()
                fig, ax = plt.subplots(figsize=(8,4))
                if not piv.empty:
                    x = np.arange(len(piv.index))
                    width = 0.35
                    yL = piv["LONG"].values if "LONG" in piv.columns else np.zeros(len(x))
                    yS = piv["SHORT"].values if "SHORT" in piv.columns else np.zeros(len(x))
                    ax.bar(x - width/2, yL, width, label="LONG")
                    ax.bar(x + width/2, yS, width, label="SHORT")
                    ax.set_xticks(x); ax.set_xticklabels([f"RR {int(i)}" for i in piv.index])
                else:
                    ax.bar(rr["rr_used"].astype(str)+"-"+rr["side"], rr["pnl"])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_ylabel("Total PnL"); ax.set_title("EMA — PnL by RR & Side"); ax.legend()
                st.pyplot(fig, use_container_width=True)

                eq = bt_ema["equity"]
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(eq.index, eq.values, label="Equity")
                ax2.set_title(f"EMA — Equity Curve (Final: {eq.iloc[-1]:,.0f})")
                ax2.grid(alpha=0.3); ax2.legend()
                st.pyplot(fig2, use_container_width=True)

            st.subheader("Tables — EMA")
            if not bt_ema["log"].empty:
                show_cols = ["side","entry_date","exit_date","rr_used","lot","entry","sl_prev_bar","exit","pnl","equity_after"]
                st.dataframe(bt_ema["log"][show_cols], use_container_width=True, height=320)
                st.dataframe(
                    bt_ema["per_year"][["year","side","trades","wins","winrate%","pnl","pnl_pct_of_capital"]],
                    use_container_width=True
                )
            else:
                st.info("No EMA trades to display.")

        # --------------- EMA+SMC TAB ----------------
        with tabs[1]:
            st.subheader("Metrics — EMA+SMC")
            st.json({k: (None if pd.isna(v) else round(float(v), 4)) for k, v in mets_smc.items()})

            st.subheader("Forecast Plan — EMA+SMC")
            planL = latest_plan_long(df_smc); planS = latest_plan_short(df_smc)
            c = st.columns(5)
            if planL:
                c[0].metric("Long Entry", planL["entry"]); c[1].metric("Long SL", planL["sl"])
                c[2].metric("TP1", planL["tp1"]); c[3].metric("TP2", planL["tp2"]); c[4].metric("TP3", planL["tp3"])
                st.caption(f"Long risk distance: {planL['risk_dist']}")
            if planS:
                c2 = st.columns(5)
                c2[0].metric("Short Entry", planS["entry"]); c2[1].metric("Short SL", planS["sl"])
                c2[2].metric("TP1", planS["tp1"]); c2[3].metric("TP2", planS["tp2"]); c2[4].metric("TP3", planS["tp3"])
                st.caption(f"Short risk distance: {planS['risk_dist']}")

            st.subheader("Backtest Charts — EMA+SMC")
            rr2 = bt_smc["rr_perf"]
            if rr2.empty:
                st.warning("EMA+SMC: No trades. Try wider dates, lower threshold, or different symbol.")
            else:
                # PnL by RR & side
                try:
                    piv2 = rr2.pivot(index="rr_used", columns="side", values="pnl").fillna(0.0)
                except Exception:
                    piv2 = pd.DataFrame()
                fig, ax = plt.subplots(figsize=(8,4))
                if not piv2.empty:
                    x = np.arange(len(piv2.index))
                    width = 0.35
                    yL = piv2["LONG"].values if "LONG" in piv2.columns else np.zeros(len(x))
                    yS = piv2["SHORT"].values if "SHORT" in piv2.columns else np.zeros(len(x))
                    ax.bar(x - width/2, yL, width, label="LONG")
                    ax.bar(x + width/2, yS, width, label="SHORT")
                    ax.set_xticks(x); ax.set_xticklabels([f"RR {int(i)}" for i in piv2.index])
                else:
                    ax.bar(rr2["rr_used"].astype(str)+"-"+rr2["side"], rr2["pnl"])
                ax.set_ylabel("Total PnL"); ax.set_title("EMA+SMC — PnL by RR & Side"); ax.legend()
                st.pyplot(fig, use_container_width=True)

                eq2 = bt_smc["equity"]
                fig2, ax2 = plt.subplots(figsize=(10,4))
                ax2.plot(eq2.index, eq2.values, label="Equity")
                ax2.set_title(f"EMA+SMC — Equity Curve (Final: {eq2.iloc[-1]:,.0f})")
                ax2.grid(alpha=0.3); ax2.legend()
                st.pyplot(fig2, use_container_width=True)

            st.subheader("Tables — EMA+SMC")
            if not bt_smc["log"].empty:
                show_cols = ["side","entry_date","exit_date","rr_used","lot","entry","sl_prev_bar","exit","pnl","equity_after"]
                st.dataframe(bt_smc["log"][show_cols], use_container_width=True, height=320)
                st.dataframe(
                    bt_smc["per_year"][["year","side","trades","wins","winrate%","pnl","pnl_pct_of_capital"]],
                    use_container_width=True
                )
            else:
                st.info("No EMA+SMC trades to display.")

    except Exception as e:
        st.error(f"Run failed: {e}")
else:
    st.info("Set parameters in the sidebar and click **Run**.")

