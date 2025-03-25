import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Portfolio Risk Analyzer 📊")
st.markdown("""
This tool helps you analyze your investment portfolio's risk metrics and run various scenario analyses.
Upload your portfolio or generate a random one to get started!
""")

# Cache the data loading
@st.cache_data
def load_monthly_returns():
    try:
        return pd.read_csv("monthly_returns.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error("monthly_returns.csv not found. Please ensure the file exists in the application directory.")
        return None

@st.cache_data
def generate_portfolio_template():
    template_df = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'VTI', 'BND', 'GLD'],
        'Weight': [0.25, 0.25, 0.20, 0.15, 0.15]
    })
    return template_df

def calculate_backtracked_weights(portfolio_df, returns_data, lookback_years):
    """Calculate weights backtracked to a previous point in time"""
    # Get the tickers
    tickers = portfolio_df['Ticker'].tolist()
    current_weights = portfolio_df.set_index('Ticker')['Weight']
    
    # Get returns for the lookback period
    if not isinstance(returns_data.index, pd.DatetimeIndex):
        returns_data.index = pd.to_datetime(returns_data.index)
    
    # Calculate how many months to look back
    lookback_months = lookback_years * 12
    
    # If we don't have enough data, use what we have
    max_lookback = min(lookback_months, len(returns_data))
    
    # Get the cumulative returns for each ticker for the lookback period
    ticker_returns = returns_data[tickers].iloc[-max_lookback:]
    cumulative_returns = (1 + ticker_returns).cumprod()
    
    # The returns from now back to the start
    total_returns = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) - 1
    
    # Adjust weights based on returns
    # If a stock had a 50% return, its weight at the starting point would be current_weight / 1.5
    backtracked_weights = {}
    for ticker in tickers:
        # Add 1 to get the total return factor
        return_factor = 1 + total_returns[ticker]
        # Avoid division by zero or negative numbers
        if return_factor <= 0:
            backtracked_weights[ticker] = 0
        else:
            backtracked_weights[ticker] = current_weights[ticker] / return_factor
    
    # Normalize the weights to sum to 1
    weights_sum = sum(backtracked_weights.values())
    normalized_weights = {ticker: weight / weights_sum for ticker, weight in backtracked_weights.items()}
    
    return normalized_weights

def calculate_portfolio_value_over_time(portfolio_df, returns_data, lookback_years):
    """Calculate portfolio value over time with backtracked weights"""
    # Get the tickers
    tickers = portfolio_df['Ticker'].tolist()
    
    # Calculate backtracked weights
    backtracked_weights = calculate_backtracked_weights(portfolio_df, returns_data, lookback_years)
    
    # Calculate how many months to look back
    lookback_months = lookback_years * 12
    
    # If we don't have enough data, use what we have
    max_lookback = min(lookback_months, len(returns_data))
    
    # Get returns for the lookback period
    ticker_returns = returns_data[tickers].iloc[-max_lookback:]
    
    # Convert the backtracked weights to a series with ticker as index
    backtracked_weights_series = pd.Series(backtracked_weights)
    
    # Calculate weighted returns for each period
    weighted_returns = ticker_returns.mul(backtracked_weights_series, axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)
    
    # Calculate portfolio value over time (starting at 100)
    portfolio_value = 100 * (1 + portfolio_returns).cumprod()
    
    return portfolio_value, portfolio_returns, backtracked_weights

def calculate_portfolio_metrics(portfolio_df, returns_data, lookback_years):
    """Calculate key portfolio risk metrics"""
    # Calculate portfolio value and returns over time
    portfolio_value, portfolio_returns, backtracked_weights = calculate_portfolio_value_over_time(
        portfolio_df, returns_data, lookback_years
    )
    
    # Convert backtracked weights to dataframe for display
    backtracked_weights_df = pd.DataFrame(list(backtracked_weights.items()), 
                                         columns=['Ticker', f'Weight (T-{lookback_years})'])
    
    # Annual return based on total return over the period
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    annual_return = (1 + total_return) ** (1 / lookback_years) - 1
    
    # Calculate other metrics based on current weights
    # For volatility, use the current weights and recent returns
    current_weights = portfolio_df.set_index('Ticker')['Weight']
    tickers = portfolio_df['Ticker'].tolist()
    
    # Get recent returns (last 12 months) for volatility calculation
    recent_returns = returns_data[tickers].iloc[-12:]
    
    # Calculate weighted returns with current weights
    current_weighted_returns = recent_returns.mul(current_weights, axis=1)
    current_portfolio_returns = current_weighted_returns.sum(axis=1)
    
    # Annual volatility based on current weights
    annual_vol = np.std(current_portfolio_returns) * np.sqrt(12)
    
    # Calculate other metrics based on the portfolio returns over time
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    var_95 = calculate_historical_var(portfolio_returns, 0.95)
    var_99 = calculate_historical_var(portfolio_returns, 0.99)
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'VaR (99%)': var_99,
        'Portfolio Value': portfolio_value,
        'Backtracked Weights': backtracked_weights_df
    }

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    annual_return = np.mean(returns) * 12
    annual_volatility = np.std(returns) * np.sqrt(12)
    return (annual_return - risk_free_rate) / annual_volatility

def calculate_max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_historical_var(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int(len(sorted_returns) * (1 - confidence_level))
    return sorted_returns[index]

def validate_portfolio(portfolio_df, returns_data):
    """Validate portfolio data"""
    errors = []
    
    # Check if weights sum to 1
    weight_sum = portfolio_df['Weight'].sum()
    if not 0.99 <= weight_sum <= 1.01:
        errors.append(f"Portfolio weights sum to {weight_sum:.2f}, should be close to 1.0")
    
    # Check if all tickers exist in returns data
    missing_tickers = [ticker for ticker in portfolio_df['Ticker'] if ticker not in returns_data.columns]
    if missing_tickers:
        errors.append(f"Missing tickers in historical data: {', '.join(missing_tickers)}")
    
    return errors

def calculate_beta(asset_returns, market_returns):
    """
    Calculate the beta of an asset relative to the market using proper data alignment
    and consistent statistical methods.
    
    Beta = Covariance(asset, market) / Variance(market)
    """
    # Ensure data is aligned and drop any missing values
    combined = pd.DataFrame({'asset': asset_returns, 'market': market_returns})
    combined = combined.dropna()
    
    if len(combined) == 0:
        return np.nan
    
    # Extract aligned series
    asset = combined['asset']
    market = combined['market']
    
    # Calculate beta using aligned data with consistent degrees of freedom
    covariance = asset.cov(market)
    variance = market.var()
    
    return covariance / variance

# Sidebar navigation with only two options now
st.sidebar.title("Portfolio Input")

# Load historical returns data
returns_data = load_monthly_returns()

if returns_data is None:
    st.stop()

# Section 1: Portfolio Input (moved to sidebar)
input_method = st.sidebar.radio("Choose input method", ["Generate Random Portfolio", "Upload File", "Enter Manually"])

if input_method == "Generate Random Portfolio":
    if st.sidebar.button("Generate Random Portfolio"):
        # Select random tickers - ensure at least 10 stocks
        num_assets = np.random.randint(10, 16)  # Between 10 and 15 assets
        selected_tickers = np.random.choice(returns_data.columns, num_assets, replace=False)
        
        # Generate random weights and normalize
        weights = np.random.random(num_assets)
        weights = weights / weights.sum()
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame({
            'Ticker': selected_tickers,
            'Weight': weights
        })
        
        st.session_state['portfolio'] = portfolio_df
        st.sidebar.success("Random portfolio generated!")

elif input_method == "Upload File":
    # Download template button
    template_df = generate_portfolio_template()
    buffer = io.BytesIO()
    template_df.to_csv(buffer, index=False)
    st.sidebar.download_button(
        label="Download Template CSV",
        data=buffer.getvalue(),
        file_name="portfolio_template.csv",
        mime="text/csv"
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your portfolio CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                portfolio_df = pd.read_csv(uploaded_file)
            else:
                portfolio_df = pd.read_excel(uploaded_file)
            
            # Validate portfolio
            errors = validate_portfolio(portfolio_df, returns_data)
            if errors:
                for error in errors:
                    st.sidebar.error(error)
            else:
                st.session_state['portfolio'] = portfolio_df
                st.sidebar.success("Portfolio uploaded successfully!")
                
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")

else:  # Enter Manually
    st.sidebar.subheader("Enter Portfolio Manually")
    
    # Initialize session state for manual portfolio if not exists
    if 'manual_portfolio_rows' not in st.session_state:
        st.session_state['manual_portfolio_rows'] = 3  # Start with 3 rows
        
    # Function to add a row
    def add_row():
        st.session_state['manual_portfolio_rows'] += 1
    
    # Button to add more rows
    st.sidebar.button("Add Row", on_click=add_row)
    
    # Get available tickers from returns data
    available_tickers = returns_data.columns.tolist()
    
    # Create lists to store ticker and weight inputs
    tickers = []
    weights = []
    
    # Create form for manual entry
    with st.sidebar.form("manual_portfolio_form"):
        st.write("Enter tickers and weights (in %):")
        
        # Create rows for ticker and weight inputs
        for i in range(st.session_state['manual_portfolio_rows']):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Searchable ticker input with dropdown
                ticker = st.selectbox(
                    f"Ticker {i+1}",
                    options=available_tickers,
                    key=f"ticker_{i}"
                )
                tickers.append(ticker)
            
            with col2:
                # Weight input (as percentage)
                weight = st.number_input(
                    f"Weight {i+1} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    key=f"weight_{i}"
                )
                weights.append(weight / 100.0)  # Convert percentage to decimal
        
        # Check total and build portfolio on submit
        submit_button = st.form_submit_button("Create Portfolio")
        
        if submit_button:
            # Filter out empty entries (where weight is 0)
            portfolio_data = [(t, w) for t, w in zip(tickers, weights) if w > 0]
            
            if not portfolio_data:
                st.error("Please add at least one position with weight greater than 0.")
            else:
                # Extract tickers and weights
                filtered_tickers, filtered_weights = zip(*portfolio_data)
                
                # Check if weights sum to 1 (100%)
                total_weight = sum(filtered_weights)
                
                if not 0.99 <= total_weight <= 1.01:
                    st.error(f"Total weight is {total_weight:.2%}, must be 100%.")
                else:
                    # Create portfolio DataFrame
                    portfolio_df = pd.DataFrame({
                        'Ticker': filtered_tickers,
                        'Weight': filtered_weights
                    })
                    
                    # Normalize weights to ensure they sum exactly to 1
                    portfolio_df['Weight'] = portfolio_df['Weight'] / portfolio_df['Weight'].sum()
                    
                    # Validate tickers exist in returns data
                    missing_tickers = [ticker for ticker in portfolio_df['Ticker'] if ticker not in returns_data.columns]
                    if missing_tickers:
                        st.error(f"Missing tickers in historical data: {', '.join(missing_tickers)}")
                    else:
                        st.session_state['portfolio'] = portfolio_df
                        st.success("Portfolio created successfully!")

# Add Help section to sidebar
with st.sidebar.expander("Help & Documentation"):
    st.write("""
    **Metric Definitions**
    - Annual Return: The yearly return of the portfolio
    - Annual Volatility: Standard deviation of returns
    - Sharpe Ratio: Risk-adjusted return measure
    - Max Drawdown: Maximum observed loss
    - VaR: Minimum expected loss at a confidence level
    
    **Usage Instructions**
    1. Upload portfolio or generate random one
    2. Review the portfolio analysis
    3. Use scenario testing to simulate events
    """)

# Main content area - Portfolio Analysis
if 'portfolio' in st.session_state:
    st.header("Portfolio Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Portfolio Weights")
        st.dataframe(st.session_state['portfolio'])
        
        # Display portfolio composition pie chart
        # Sort the portfolio by weight in descending order
        sorted_portfolio = st.session_state['portfolio'].sort_values(by='Weight', ascending=False)
        fig = px.pie(sorted_portfolio, values='Weight', names='Ticker', title='Portfolio Composition')
        # Modify the figure to ensure clockwise direction
        fig.update_traces(rotation=0, direction='clockwise')
        st.plotly_chart(fig)
    
    # Add lookback period selector
    lookback_years = st.selectbox("Select Analysis Period", [1, 3, 5], index=0)
    
    # Calculate and display portfolio value over time
    try:
        metrics = calculate_portfolio_metrics(st.session_state['portfolio'], returns_data, lookback_years)
        
        # Display portfolio value chart
        st.subheader(f"Portfolio Value (Past {lookback_years} Year{'s' if lookback_years > 1 else ''})")
        
        # Check if SPY exists in the data for comparison
        if 'SPY' in returns_data.columns:
            # Get SPY returns for the same period
            spy_returns = returns_data['SPY'].iloc[-len(metrics['Portfolio Value']):]
            
            # Calculate SPY value (also starting at 100)
            spy_value = 100 * (1 + spy_returns).cumprod()
            
            # Create DataFrame with both portfolio and SPY values
            comparison_df = pd.DataFrame({
                'Portfolio': metrics['Portfolio Value'].values,
                'SPY': spy_value.values
            }, index=metrics['Portfolio Value'].index)
            
            # Plot both lines
            fig = px.line(
                comparison_df,
                labels={"value": "Value", "variable": ""},
                title=f"Portfolio vs. SPY (Starting at 100)"
            )
            st.plotly_chart(fig)
        else:
            # Plot only portfolio if SPY is not available
            fig = px.line(
                x=metrics['Portfolio Value'].index,
                y=metrics['Portfolio Value'].values,
                labels={"x": "Date", "y": "Value"},
                title=f"Portfolio Value (Starting at 100)"
            )
            st.plotly_chart(fig)
        
        # Display backtracked weights
        st.subheader(f"Backtracked Weights ({lookback_years} Year{'s' if lookback_years > 1 else ''} Ago)")
        st.dataframe(metrics['Backtracked Weights'])
        
        # Display risk metrics
        st.subheader("Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Annual Return ({lookback_years}Y)", f"{metrics['Annual Return']:.2%}")
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        with col2:
            st.metric("Annual Volatility (Current)", f"{metrics['Annual Volatility']:.2%}")
            st.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            st.metric("VaR (99%)", f"{metrics['VaR (99%)']:.2%}")
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        portfolio_returns = returns_data[st.session_state['portfolio']['Ticker']]
        correlation_method = st.selectbox("Correlation Method", ["pearson", "spearman"])
        correlation_matrix = portfolio_returns.corr(method=correlation_method)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.index,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(title=f"{correlation_method.capitalize()} Correlation Matrix")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")
    
    # Section 3: Scenario Testing
    st.header("Scenario Testing")
    
    # Market drop simulation
    st.subheader("Market Drop Simulation")
    market_drop = st.slider("Market Drop Percentage", -50, -5, -20, 5)
    
    # Calculate portfolio beta relative to SPY
    if 'SPY' in returns_data.columns:
        market_returns = returns_data['SPY']
        
        betas = {}
        for ticker in st.session_state['portfolio']['Ticker']:
            # Use the improved beta calculation function
            betas[ticker] = calculate_beta(returns_data[ticker], market_returns)
        
        # Calculate expected portfolio impact
        impact = sum(
            st.session_state['portfolio'].set_index('Ticker')['Weight'] * 
            pd.Series(betas) * 
            market_drop / 100
        )
        
        st.metric("Expected Portfolio Impact", f"{impact:.2%}")
        
        # Display beta table
        beta_df = pd.DataFrame(list(betas.items()), columns=['Ticker', 'Beta'])
        # Merge with weights
        beta_df = beta_df.merge(st.session_state['portfolio'], on='Ticker')
        beta_df['Contribution to Impact'] = beta_df['Beta'] * beta_df['Weight'] * market_drop / 100
        
        st.subheader("Asset Betas and Impact")
        st.dataframe(beta_df)
    else:
        st.warning("Market index (SPY) data not available for beta calculation")
    
    # Historical event replay
    st.subheader("Historical Event Replay")
    events = {
        "COVID-19 Crash (Jan-Apr 2020)": ("01/01/2020", "04/30/2020"),
        "2022 Bear Market (Jan-Oct 2022)": ("01/01/2022", "10/31/2022"),
    }
    
    selected_event = st.selectbox("Select Historical Event", list(events.keys()))

    # Load daily returns for historical events
    @st.cache_data
    def load_daily_returns():
        try:
            return pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
        except FileNotFoundError:
            st.error("daily_returns.csv not found. Please ensure the file exists in the application directory.")
            return None

    daily_returns_data = load_daily_returns()
    
    if daily_returns_data is None:
        st.stop()
    
    # Check if we have data for the selected event
    try:
        start_date_str, end_date_str = events[selected_event]
        start_date = datetime.strptime(start_date_str, "%m/%d/%Y")
        end_date = datetime.strptime(end_date_str, "%m/%d/%Y")
        
        # Check if we have SPY or VOO for market returns
        market_ticker = None
        if 'SPY' in daily_returns_data.columns:
            market_ticker = 'SPY'
        elif 'VOO' in daily_returns_data.columns:
            market_ticker = 'VOO'
        
        if market_ticker is None:
            st.warning("Market index (SPY or VOO) data not available for historical analysis")
        else:
            # Get market returns for the period
            market_data = daily_returns_data[[market_ticker]]
            
            # Convert index to datetime if it's not already
            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data.index = pd.to_datetime(market_data.index)
            
            # Filter data for the period
            period_data = market_data[(market_data.index >= start_date) & 
                                    (market_data.index <= end_date)]
            
            if len(period_data) == 0:
                st.warning(f"No historical data available for {selected_event}")
            else:
                # Calculate portfolio performance for the period
                tickers = st.session_state['portfolio']['Ticker'].tolist()
                
                # Get all tickers' returns for the period
                tickers_data = daily_returns_data[tickers]
                if not isinstance(tickers_data.index, pd.DatetimeIndex):
                    tickers_data.index = pd.to_datetime(tickers_data.index)
                
                period_tickers_data = tickers_data[(tickers_data.index >= start_date) & 
                                                  (tickers_data.index <= end_date)]
                
                # Calculate weighted returns for the portfolio
                weighted_returns = period_tickers_data.mul(
                    st.session_state['portfolio'].set_index('Ticker')['Weight'], 
                    axis=1
                )
                portfolio_returns = weighted_returns.sum(axis=1)
                
                # Calculate cumulative returns
                portfolio_cum_return = (1 + portfolio_returns).cumprod()
                market_cum_return = (1 + period_data[market_ticker]).cumprod()
                
                # Calculate maximum drawdown for both portfolio and market
                portfolio_peak = portfolio_cum_return.expanding(min_periods=1).max()
                portfolio_drawdown = (portfolio_cum_return - portfolio_peak) / portfolio_peak
                portfolio_max_drawdown = portfolio_drawdown.min()
                
                market_peak = market_cum_return.expanding(min_periods=1).max()
                market_drawdown = (market_cum_return - market_peak) / market_peak
                market_max_drawdown = market_drawdown.min()
                
                # Display metrics
                st.metric(f"Portfolio Maximum Drawdown", f"{portfolio_max_drawdown:.2%}")
                st.metric(f"{market_ticker} Maximum Drawdown", f"{market_max_drawdown:.2%}")
                
                # Create DataFrame with both market and portfolio performance
                comparison_df = pd.DataFrame({
                    'Portfolio': portfolio_cum_return.values,
                    market_ticker: market_cum_return.values
                }, index=period_data.index)
                
                # Display a line chart with both lines
                st.subheader(f"Performance during {selected_event}")
                fig = px.line(
                    comparison_df,
                    labels={"value": "Cumulative Return", "variable": ""},
                    title=f"Portfolio vs. {market_ticker}"
                )
                st.plotly_chart(fig)
                
                # Display drawdown chart
                drawdown_df = pd.DataFrame({
                    'Portfolio Drawdown': portfolio_drawdown.values,
                    f'{market_ticker} Drawdown': market_drawdown.values
                }, index=period_data.index)
                
                fig_drawdown = px.line(
                    drawdown_df,
                    labels={"value": "Drawdown", "variable": ""},
                    title="Drawdown Analysis"
                )
                fig_drawdown.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig_drawdown)
                
    except (KeyError, ValueError) as e:
        st.warning(f"Error analyzing event data: {str(e)}")
else:
    st.header("Welcome to Portfolio Risk Analyzer")
    st.info("Please input your portfolio using the options in the sidebar to begin analysis.") 