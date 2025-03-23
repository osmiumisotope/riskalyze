# Portfolio Risk Analyzer

A Streamlit-based web application for analyzing investment portfolio risk metrics and running scenario analyses.

## Features

- Portfolio input via CSV/Excel upload or random generation
- Risk metrics calculation (returns, volatility, Sharpe ratio, etc.)
- Interactive visualizations
- Scenario testing with market drop simulation
- Correlation analysis with multiple methods
- Educational components with metric definitions

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Create a `monthly_returns.csv` file with historical returns data
   - Format: First column should be dates, subsequent columns should be ticker symbols
   - Values should be decimal returns (e.g., 0.05 for 5% return)

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically http://localhost:8501)

3. Use the application:
   - Upload your portfolio CSV/Excel file or generate a random portfolio
   - Navigate through different sections using the sidebar
   - View risk metrics and visualizations
   - Test different market scenarios

## Portfolio File Format

Your portfolio file should contain two columns:
- Ticker: Stock/ETF symbol (e.g., "AAPL", "SPY")
- Weight: Decimal allocation (e.g., 0.25 for 25%)

Example:
```csv
Ticker,Weight
AAPL,0.25
MSFT,0.25
VTI,0.20
BND,0.15
GLD,0.15
```

## Data Requirements

The `monthly_returns.csv` file should contain:
- First row: Header with ticker symbols
- First column: Dates in mm/dd/yyyy format
- All other cells: Monthly return values as decimals 