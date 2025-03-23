import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time


def download_monthly_data(tickers, start_date, end_date, interval='1mo'):
    """
    Download monthly data for a list of tickers between start_date and end_date.
    Uses yfinance's batch download to reduce the number of requests.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by='ticker',
        auto_adjust=True,  # returns adjusted prices in the 'Close' column
        threads=True
    )
    return data


def extract_adjusted_close(data, tickers):
    """
    Extract adjusted close prices from the downloaded data.
    Handles both multi-ticker (with MultiIndex columns) and single-ticker data.
    """
    if isinstance(data.columns, pd.MultiIndex):
        # If available, prefer 'Adj Close', otherwise use 'Close'
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"]
        else:
            prices = pd.DataFrame(
                {ticker: data[ticker]['Close'] for ticker in tickers if 'Close' in data[ticker].columns})
    else:
        if 'Adj Close' in data.columns:
            prices = data[['Adj Close']]
        elif 'Close' in data.columns:
            prices = data[['Close']]
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in data.")
    return prices


def compute_monthly_returns(prices):
    """
    Compute monthly returns as percentage changes from the adjusted close prices.
    """
    returns = prices.pct_change().dropna()
    return returns


def main():
    # Define a list of commonly used ETFs and stocks.
    tickers = [
        'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'VTI', 'AGG',
        'LQD', 'IEF', 'TIP', 'XLF', 'XLY', 'XLV', 'XLI', 'XLE', 'XLB', 'GDX',
        'SLV', 'USO', 'GLD', 'UUP', 'VXX',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM', 'V',
        'PG', 'UNH', 'NVDA', 'HD', 'DIS', 'BAC', 'MA', 'XOM', 'VZ', 'KO',
        'PFE', 'INTC', 'T', 'CSCO', 'CVX', 'WMT', 'MRK', 'NKE', 'IBM', 'MCD',
        'ORCL', 'ABT', 'CRM', 'PEP', 'ADBE', 'QCOM', 'LLY', 'MDT', 'UPS', 'DHR',
        'TXN', 'C', 'BA', 'GE', 'HON', 'AMGN', 'BMY', 'CAT', 'SBUX', 'COST',
        'MMM', 'GS', 'CVS', 'BLK', 'DE', 'AMT', 'NOW', 'BKNG', 'SPGI',
        'F', 'GM', 'TSM', 'AMD', 'XOP', 'VUG', 'VTV', 'IWD', 'IWF',
        'PYPL', 'ADP', 'USB', 'BK', 'EMB', 'NFLX'
    ]

    # Define the date range: 10 years from today.
    end_date = datetime.today().date()
    start_date = (datetime.today() - relativedelta(years=10)).date()

    print(f"Downloading monthly data for {len(tickers)} tickers from {start_date} to {end_date}...")

    # Batch download data.
    data = download_monthly_data(tickers, start_date, end_date, interval='1mo')

    # Extract adjusted close prices.
    prices = extract_adjusted_close(data, tickers)

    # Compute monthly returns.
    monthly_returns = compute_monthly_returns(prices)

    # Save the data to CSV files.
    prices.to_csv("monthly_prices.csv")
    monthly_returns.to_csv("monthly_returns.csv")

    print("Download and processing complete.")
    print("Data saved as 'monthly_prices.csv' and 'monthly_returns.csv'.")

    # For continuous updates, consider integrating a scheduler instead of a simple sleep.
    # time.sleep(3600)  # e.g., sleep for 1 hour to stay within rate limits


if __name__ == "__main__":
    main()
