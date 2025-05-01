import pandas as pd
from pathlib import Path
import yfinance as yf
import time

# ------------------- Part 1: Scraping Tickers (making the list of tickers) -------------------
wiki = "https://en.wikipedia.org/wiki/Nasdaq-100" # Link to wikipedia article that contains table with list of QQQ companies
try:
    tickers = (
        pd.read_html(wiki, match="Ticker")[0] # only collecting the data from the ticker column - the ticker names
        .Ticker.dropna().unique().tolist()
    )
    print("Tickers successfully scraped:")
    print(tickers)
except Exception as e:
    print(f"Error scraping tickers: {e}")
    tickers = [] # Initialize as empty list to avoid errors later

# ------------------- Part 2: Downloading Data -------------------
START = "2015-04-21" # setting the start date of the market data I'm collecting to 10 years ago
DATA_DIR = Path("data/qqq_dfs")  # New directory for DataFrames
DATA_DIR.mkdir(parents=True, exist_ok=True)
SLEEP = 1  # creating a one second pause between calls so I don't get throttled
# ------------------------------

all_dfs = {}  # Dictionary to store DataFrames

if tickers: # Only proceed if the tickers list is not empty
    for tkr in tickers:
        try:
            df = yf.download(
                tickers=tkr,
                start=START,
                progress=False,
                auto_adjust=True
            )
            if df.empty:
                print(f"{tkr}: no data")
                continue
            all_dfs[tkr] = df  # Store the DataFrame in the dictionary
            print(f"✔  {tkr} DataFrame stored in memory")
            # If you also want to save to a different format (e.g., Pickle):
            # out_file = DATA_DIR / f"{tkr}.pkl"
            # df.to_pickle(out_file)
            # print(f"✔  {tkr} saved as Pickle → {out_file}")
        except Exception as e:
            print(f"✖  {tkr}: {e}")
        time.sleep(SLEEP)

    # The 'all_dfs' dictionary now contains all the downloaded DataFrames,
    # with the ticker symbol as the key.
    print("\nAll DataFrames downloaded and stored in the 'all_dfs' dictionary.")
else:
    print("No tickers found. Skipping data download.")