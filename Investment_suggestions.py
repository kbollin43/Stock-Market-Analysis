import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Helper functions
def calculate_volatility(returns):
    return returns.std() * np.sqrt(252)  # Annualized volatility

def portfolio_optimization(stock_data, stock_symbols, risk_free_rate=0.02):
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_assets = len(stock_symbols)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    initial_weights = np.array([1 / num_assets] * num_assets)
    optimized = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = optimized.x

    optimized_return = np.dot(optimal_weights, mean_returns)
    optimized_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_volatility

    return optimal_weights, optimized_return, optimized_volatility, optimized_sharpe_ratio

# Updated stock categories
stocks_list = [
    'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT',
    'JPM', 'PG', 'UNH', 'MA', 'NVDA', 'HD', 'DIS', 'PYPL', 'BAC', 'VZ',
    'ADBE', 'CMCSA', 'NFLX', 'XOM', 'INTC', 'KO', 'T', 'PFE', 'CSCO', 'PEP',
    'ABBV', 'ABT', 'CRM', 'COST', 'CVX', 'NKE', 'MRK', 'MDT', 'ACN', 'TMO',
    'NEE', 'WFC', 'DHR', 'LIN', 'ORCL', 'MCD', 'AMGN', 'HON', 'QCOM', 'TXN'
]

growth_stocks = [
    'AAPL', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'NFLX', 'MSFT', 'ADBE', 'CRM',
    'PYPL', 'DIS', 'MA', 'V', 'ACN', 'ORCL'
]

income_stocks = [
    'JNJ', 'PG', 'XOM', 'T', 'VZ', 'CVX', 'PFE', 'KO', 'PEP', 'WMT', 'MCD',
    'ABBV', 'MRK', 'INTC', 'CSCO', 'AMGN', 'HD', 'DHR', 'WFC', 'TMO'
]

capital_preservation_stocks = [
    'JNJ', 'PG', 'WMT', 'KO', 'PEP', 'V', 'UNH', 'ACN', 'TMO', 'NEE',
    'LIN', 'MDT', 'ORCL', 'ABT', 'BRK-B', 'HON'
]

# Streamlit Application
st.title("Investment Strategy and Risk Mitigation")

# Investment Strategy Input
st.subheader("Investment Strategy Input")

investment_duration_years = st.number_input(
    "Enter investment duration (in years):",
    min_value=0.1,
    max_value=50.0,
    value=1.0,
    step=0.1,
    key='risk_investment_duration'
)

risk_appetite = st.selectbox(
    "Select your risk appetite:",
    options=["Low", "Medium", "High"],
    key='risk_risk_appetite'
)

investing_goals = st.multiselect(
    "Select your investing goals:",
    options=["Growth", "Income", "Capital Preservation"],
    key='risk_investing_goals'
)

investment_horizon = st.selectbox(
    "Select your investment horizon:",
    options=["Short Term", "Long Term"],
    key='risk_investment_horizon'
)

# Map investing goals to specific stocks
goal_stocks = set()
if 'Growth' in investing_goals:
    goal_stocks.update(growth_stocks)
if 'Income' in investing_goals:
    goal_stocks.update(income_stocks)
if 'Capital Preservation' in investing_goals:
    goal_stocks.update(capital_preservation_stocks)

# Download stock data based on investment duration
start_date = date.today() - timedelta(days=int(investment_duration_years * 365))
end_date = date.today()

try:
    stock_data = yf.download(stocks_list, start=start_date, end=end_date)['Adj Close']
    if stock_data.empty:
        st.error("No data available for the selected stocks.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching stock data: {str(e)}")
    st.stop()

# Calculate returns and volatility
returns = stock_data.pct_change().dropna()
volatility = returns.std() * np.sqrt(252)
volatility_df = volatility.reset_index()
volatility_df.columns = ['Stock', 'Volatility']

# Filter stocks based on risk appetite
if risk_appetite == 'Low':
    filtered_stocks = volatility_df[volatility_df['Volatility'] < 0.20]['Stock'].tolist()
elif risk_appetite == 'Medium':
    filtered_stocks = volatility_df[(volatility_df['Volatility'] >= 0.20) & (volatility_df['Volatility'] <= 0.35)]['Stock'].tolist()
else:  # High risk appetite
    filtered_stocks = volatility_df[volatility_df['Volatility'] > 0.35]['Stock'].tolist()

# Further filter stocks based on investing goals
if investing_goals:
    filtered_stocks = list(set(filtered_stocks).intersection(goal_stocks))

# Display filtered stocks
if not filtered_stocks:
    st.error("No stocks match your criteria. Please adjust your risk appetite or investing goals.")
else:
    st.write(f"Based on your inputs, the following stocks match your criteria: {', '.join(filtered_stocks)}")

    # Perform portfolio optimization
    if len(filtered_stocks) >= 2:
        portfolio_data = stock_data[filtered_stocks]
        optimal_weights, optimized_return, optimized_volatility, optimized_sharpe_ratio = portfolio_optimization(
            portfolio_data, filtered_stocks, risk_free_rate=0.02
        )

        st.subheader("Recommended Portfolio Allocation:")
        for stock, weight in zip(filtered_stocks, optimal_weights):
            st.write(f"{stock}: {weight * 100:.2f}%")
        st.write(f"Expected Annual Return: {optimized_return:.2%}")
        st.write(f"Expected Annual Volatility: {optimized_volatility:.2%}")
        st.write(f"Expected Sharpe Ratio: {optimized_sharpe_ratio:.2f}")
    else:
        st.subheader("Recommended Stock:")
        st.write(f"We recommend investing in {filtered_stocks[0]} based on your criteria.")