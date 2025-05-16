import tkinter as tk
from tkinter import ttk, messagebox
from sentimentMapping import black_scholes_greeks, iv_adjust, get_sentiment_full
from newspaper import Article
import yfinance as yf
from datetime import datetime
import finalAnalysis

class GreeksApp:
    def __init__(self, root):
        self.root = root
        root.title("Dynamic Greeks Calibrator")
        
        # Input fields
        ttk.Label(root, text="Company Ticker (e.g. AAPL)").grid(row=0, column=0, sticky=tk.W)
        self.ticker_entry = ttk.Entry(root)
        self.ticker_entry.grid(row=0, column=1)

        ttk.Label(root, text="Strike Price").grid(row=1, column=0, sticky=tk.W)
        self.strike_entry = ttk.Entry(root)
        self.strike_entry.grid(row=1, column=1)

        ttk.Label(root, text="Expiry (YYYY-MM-DD)").grid(row=2, column=0, sticky=tk.W)
        self.expiry_entry = ttk.Entry(root)
        self.expiry_entry.grid(row=2, column=1)

        ttk.Label(root, text="Implied Volatility (e.g. 0.25)").grid(row=3, column=0, sticky=tk.W)
        self.iv_entry = ttk.Entry(root)
        self.iv_entry.grid(row=3, column=1)

        ttk.Label(root, text="Risk-Free Rate (e.g. 0.05)").grid(row=4, column=0, sticky=tk.W)
        self.r_entry = ttk.Entry(root)
        self.r_entry.grid(row=4, column=1)

        ttk.Label(root, text="Call or Put (C/P)").grid(row=5, column=0, sticky=tk.W)
        self.cp_entry = ttk.Entry(root)
        self.cp_entry.grid(row=5, column=1)

        ttk.Label(root, text="News Article URL").grid(row=6, column=0, sticky=tk.W)
        self.url_entry = ttk.Entry(root, width=40)
        self.url_entry.grid(row=6, column=1)

        ttk.Button(root, text="Calibrate", command=self.calibrate).grid(row=7, column=0, columnspan=2, pady=10)

    def calibrate(self):
        try:
            ticker = self.ticker_entry.get().strip()
            K = float(self.strike_entry.get())
            expiry = self.expiry_entry.get().strip()
            iv = float(self.iv_entry.get())
            r = float(self.r_entry.get())
            call_put = self.cp_entry.get().strip().upper()
            url = self.url_entry.get().strip()

            # Get spot price
            tkr = yf.Ticker(ticker)
            S = tkr.history(period="1d")['Close'].iloc[0]

            # Time to expiry (fractional years)
            secs_to_exp = (datetime.fromisoformat(expiry) - datetime.utcnow()).total_seconds()
            T = secs_to_exp / (365 * 24 * 3600)

            # Get article text
            art = Article(url)
            art.download()
            art.parse()
            article = art.title + "\n" + art.text

            # Sentiment
            sent_id, conf = get_sentiment_full(article)

            # Adjust IV
            iv_new = iv_adjust(iv, sent_id, conf)

            # Recompute Greeks
            greeks = black_scholes_greeks(S, K, T, r, iv_new, call_put)

            self.show_results(iv, iv_new, greeks, call_put)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_results(self, iv_old, iv_new, greeks, call_put):
        result_win = tk.Toplevel(self.root)
        result_win.title("Calibrated Greeks and IV")
        ttk.Label(result_win, text=f"Adjusted IV: {iv_new:.4f}").pack()
        for greek, value in greeks.items():
            ttk.Label(result_win, text=f"{greek.capitalize()}: {value:.4f}").pack()
        analysis = finalAnalysis.generate_analysis(iv_new, greeks, call_put)
        ttk.Label(result_win, text=analysis, wraplength=400, justify="left").pack(pady=(10,0))
        ttk.Button(result_win, text="Close", command=result_win.destroy).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = GreeksApp(root)
    root.mainloop() 