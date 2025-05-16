# sentimentMapping.py  ── end‑to‑end demo
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
import numpy as np
from transformers import pipeline, AutoTokenizer
from newspaper import Article

# ── Black‑Scholes Greeks ───────────────────────────────────────────────
def black_scholes_greeks(S, K, T, r, vol, call_put="C"):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    sign  = 1 if call_put == "C" else -1

    delta = sign * norm.cdf(sign * d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega  = 0.01 * S * norm.pdf(d1) * np.sqrt(T)          # per 1‑vol‑pt
    theta = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T))
             - sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2)) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

# ── IV‑adjustment rule ────────────────────────────────────────────────
def iv_adjust(base_iv, sent_id, conf, k_neg=0.20, k_pos=0.10):
    if sent_id == 0:          # bearish
        return base_iv * (1 + k_neg * conf)
    elif sent_id == 2:        # bullish
        return base_iv * (1 - k_pos * conf)
    return base_iv            # neutral

# ── FinBERT pipeline & tokenizer ──────────────────────────────────────
clf = pipeline(
    "text-classification",
    model="model/finbert_finetuned",
    tokenizer="ProsusAI/finbert",
    top_k=None,                       # returns all 3 scores per call
)
tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def get_sentiment_full(text, max_len=510):
    """Chunk the article into ≤512‑token slices and aggregate sentiment."""
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [tokens[i : i + max_len - 2]          # keeps slice ≤512 incl. CLS/SEP
              for i in range(0, len(tokens), max_len - 2)]

    scores = []
    for chunk in chunks:
        txt = tok.decode(chunk)
        scores.extend(clf(txt)[0])                 # list[dict] per chunk

    # confidence‑weighted majority vote
    label_score = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for s in scores:
        label_score[s["label"]] += s["score"]

    best_label = max(label_score, key=label_score.get)
    confidence = label_score[best_label] / sum(label_score.values())
    sent_id    = clf.model.config.label2id[best_label]   # 0 / 1 / 2
    return sent_id, confidence

# ── Pull and parse a live article ─────────────────────────────────────
url = (
    "https://www.cnbc.com/2025/05/01/apple-has-managed-tariffs-so-far-"
    "says-tough-to-predict-beyond-june.html"
)
art = Article(url); art.download(); art.parse()
article = art.title + "\n" + art.text

# ── Sentiment inference ───────────────────────────────────────────────
sent_id, conf = get_sentiment_full(article)
print("FinBERT sentiment:", ["bearish", "neutral", "bullish"][sent_id],
      "conf", round(conf, 2))

# ── Option baseline data (SPY call) ───────────────────────────────────
tkr = yf.Ticker("SPY")

# first expiry at least one day away so T > 0
exp = next(e for e in tkr.options
           if (datetime.fromisoformat(e) - datetime.utcnow()).days >= 1)

# spot price
S = tkr.history(period="1d")["Close"].iloc[0]

# at‑the‑money call
calls = tkr.option_chain(exp).calls
opt   = calls.iloc[(calls["strike"] - S).abs().argmin()]

K      = opt.strike
iv_raw = opt.impliedVolatility
print(f"Picked strike {K}  expiry {exp}")

# ── Adjust IV by sentiment ────────────────────────────────────────────
iv_new = iv_adjust(iv_raw, sent_id, conf)
print("Baseline IV:", round(iv_raw, 4), "→ Adjusted IV:", round(iv_new, 4))

# ── Time to expiry (fractional years) ─────────────────────────────────
secs_to_exp = (datetime.fromisoformat(exp) - datetime.utcnow()).total_seconds()
T = secs_to_exp / (365 * 24 * 3600)

# ── Recompute Greeks ─────────────────────────────────────────────────
r = 0.05     # risk‑free rate assumption
greeks = black_scholes_greeks(S, K, T, r, iv_new)
print("Adjusted Greeks:", greeks)
