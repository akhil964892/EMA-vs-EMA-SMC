# 📈 EdgeScan — Long & Short (Prev-Bar SL) • EMA vs EMA+SMC

**EdgeScan** is a powerful Streamlit-based backtesting and forecasting tool for both **stocks and cryptocurrencies**.
It uses **machine learning (Gradient Boosting)** combined with **technical and Smart Money Concepts (SMC)** features to simulate long/short strategies using realistic stop-loss and take-profit logic.


## 🚀 Features

* 🧠 **Two ML models:**

  * **EMA Model:** Momentum-based using EMAs and volatility features.
  * **EMA + SMC Model:** Combines EMA with market structure (swing highs/lows, BOS, CHOCH, FVG, Order Blocks).

* 💹 **Automatic data sources:**

  * Stocks & indices → Yahoo Finance
  * Crypto pairs (BTC-USD, ETH-USD, etc.) → Binance API

* ⚙️ **Customizable strategy settings:**
  Holding period, lot size, SL/TP ratios, probability thresholds, and optional shorts.

* 📊 **Interactive Streamlit dashboard:**

  * Metrics (Accuracy, F1, AUC, MCC)
  * PnL charts and equity curves
  * Trade logs and yearly summaries
  * Long/short forecast plans

---

## 🧩 Installation

1. **Clone this repository** or save `app.py` to a folder.
2. Make sure you have **Python 3.9+** installed.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Install dependencies
 `requirements.txt`:
```
streamlit
pandas
numpy
scikit-learn
matplotlib
yfinance
requests
```

---

## 🖥️ Run 

In your terminal or command prompt:

```bash
streamlit run app.py
```

The app will automatically open in your browser at:
👉 [http://localhost:8501](http://localhost:8501)

---

## ⚙️ How to Use

### 1. Configure in Sidebar

| Setting             | Description                      | Example                     |
| ------------------- | -------------------------------- | --------------------------- |
| **Symbol**          | Ticker symbol (Yahoo/crypto)     | `AAPL`, `BTC-USD`, `^NSEI`  |
| **Start / End**     | Date range for backtest          | `2018-01-01` → `2025-01-01` |
| **Holding Horizon** | Days to hold position            | 5                           |
| **Fixed Lot**       | Units per trade                  | 10                          |
| **Threshold**       | Probability cutoff               | 0.6                         |
| **Enable Shorts**   | Allow short trades               | ✅                           |
| **Use Filters**     | Apply EMA or SMC context filters | ✅                           |
| **Initial Capital** | Starting capital                 | 100000                      |

Then click **Run**.

---

## 💰 Crypto Usage (Recommended)

EdgeScan is **fully compatible with crypto** — it automatically uses **Binance** for symbols like:

```
BTC-USD, ETH-USD, SOL-USD, ADA-USD, XRP-USD, DOGE-USD, BNB-USD, DOT-USD, LTC-USD, AVAX-USD, MATIC-USD
```

### ✅ Suggested Settings for Crypto Backtesting

| Setting            | Recommended                     | Reason                                    |
| ------------------ | ------------------------------- | ----------------------------------------- |
| **Threshold**      | 0.55 – 0.65                     | Balances frequency and drawdown           |
| **Horizon**        | 3 – 10 days                      | Captures short-term swings                |
| **Use Filters**    | ✅ Enabled                       | Reduces false entries                     |
| **Enable Shorts**  | ✅ Optional                      | Use if you want derivatives-style testing |
| **Symbol Example** | `BTC-USD`, `ETH-USD`, `SOL-USD` | Binance API automatically used            |

Crypto data is fetched daily (`1d` candles) with up to **1200 bars** of history per run.
No API key required.

---

## 📊 Outputs

After running, the app will show:

* **Metrics:** Accuracy, F1, AUC, MCC, Precision, Recall
* **Charts:**

  * PnL by RR (risk/reward) and side (Long/Short)
  * Equity curve with CAGR and drawdown
* **Tables:**

  * Trade logs with entry/exit/SL/TP/pnls
  * Per-year breakdowns by side
* **Forecast Plans:**

  * Next long and short setups with entry, SL, and TP1–3 levels

---

## 🧾 Example Runs

### 🪙 Crypto

```
Symbol: BTC-USD
Start: 2018-01-01
End: 2025-01-01
Threshold: 0.6
Holding Horizon: 5 days
```

## ⚠️ Notes

* Yahoo Finance may fail for some symbols — crypto pairs automatically use Binance fallback.
* Binance data covers ~3 years (daily candles).
* This is a **research/backtesting tool**, not a live trading bot.
* Fixed lot size means no compounding — ideal for testing raw strategy performance.

---
