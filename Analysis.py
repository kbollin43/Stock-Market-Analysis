import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from scipy import stats
from scipy.optimize import minimize
from fredapi import Fred
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.diagnostics import cross_validation, performance_metrics

# Helper functions
@st.cache_data
def download_stock_data(ticker):
    return yf.download(ticker)

def calculate_volatility(returns):
    return returns.std() * np.sqrt(252)  # Annualized volatility

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    volatility = calculate_volatility(returns)
    excess_returns = returns.mean() * 252 - risk_free_rate
    return excess_returns / volatility

def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def identify_trend(prices):
    # Simple trend identification using linear regression
    x = np.arange(len(prices))
    slope, _, _, _, _ = stats.linregress(x, prices)
    if slope > 0:
        return "Upward"
    elif slope < 0:
        return "Downward"
    else:
        return "Neutral"

def calculate_drawdowns(prices):
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def portfolio_optimization(stock_data, stock_symbols, risk_free_rate=0.02):
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252  # Annualized covariance matrix

    num_assets = len(stock_symbols)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # No short selling

    # Objective: Maximize Sharpe Ratio
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Negative because we want to maximize

    # Equal initial weights
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Perform optimization
    optimized = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = optimized.x

    # Calculate optimized portfolio performance
    optimized_return = np.dot(optimal_weights, mean_returns)
    optimized_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_volatility

    return optimal_weights, optimized_return, optimized_volatility, optimized_sharpe_ratio

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    data['RSI'] = rsi
    return data

def calculate_sma(data, window=30):
    data['SMA'] = data['Close'].rolling(window=window).mean()
    return data

# Monte Carlo Simulation Function
def monte_carlo_simulation():
    st.title("Monte Carlo Simulation")

    stock_ticker = st.text_input("Enter the stock ticker:", value="MSFT")
    if not stock_ticker:
        st.error("Please enter a stock ticker.")
        return

    df = download_stock_data(stock_ticker)
    if df.empty:
        st.error("Failed to download stock data.")
        return

    st.subheader(f"Stock Data for {stock_ticker}")
    st.dataframe(df)

    returns = np.log(1 + df['Adj Close'].pct_change()).dropna()  # Calculate returns
    mu, sigma = returns.mean(), returns.std()  # Mean and standard deviation of returns
    initial = df['Adj Close'].iloc[-1]  # Initial value: last adjusted closing price

    st.markdown("### Simulation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        num_days = st.text_input("Enter the number of days:", value="30")
    with col2:
        simulations = st.text_input("Enter the number of simulations:", value="100")

    if st.button("Generate"):
        try:
            num_days = int(num_days)
            simulations = int(simulations)
        except ValueError:
            st.error("Please enter valid integers for the number of days and simulations.")
            return

        max_losses = []
        all_sim_prices = []

        fig, ax = plt.subplots()

        for _ in range(simulations):
            sim_rets = np.random.normal(mu, sigma, num_days)  # Generate simulated returns
            sim_prices = initial * (sim_rets + 1).cumprod()  # Calculate simulated prices
            sim_prices = np.insert(sim_prices, 0, initial)  # Ensure the initial value is included
            max_losses.append(initial - sim_prices.min())
            all_sim_prices.append(sim_prices)
            ax.plot(sim_prices)

        largest_predicted_loss = np.max(max_losses)
        var_95 = np.percentile(max_losses, 95)
        
        st.write("Largest Predicted Loss: ${:.2f}".format(largest_predicted_loss))
        st.write("Value at Risk (95% confidence level): ${:.2f}".format(var_95))

        # Stop-Loss Recommendation
        stop_loss = initial - var_95
        st.write(f"Recommended Stop-Loss: ${stop_loss:.2f}")

        st.pyplot(fig)

        # Calculate final prices for all simulations
        final_prices = [prices[-1] for prices in all_sim_prices]
        
        # Plot histogram of final prices
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(final_prices, bins=50)
        ax_hist.set_xlabel('Final Price')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Distribution of Final Prices')
        st.pyplot(fig_hist)

# Causal Inference Function
def causal_inference():
    st.title("Causal Inference")

    stock_ticker = st.text_input("Enter the stock ticker:", value="AAPL")
    if not stock_ticker:
        st.error("Please enter a stock ticker.")
        return

    st.markdown("### Causal Inference Parameters")
    START = st.date_input("Select start date for training data", value=pd.to_datetime("2015-01-01"))
    END = st.date_input("Select end date for training data", value=pd.to_datetime("2020-01-01"))
    TODAY = date.today()
    if START >= END:
        st.error("Start date must be before end date.")
        return

    # Allow user to input FRED API key, or use default
    fred_api_key = st.text_input("Enter FRED API Key:", value="96cd73fb8e47087dd36deaf37af0ef35")
    if not fred_api_key:
        st.error("96cd73fb8e47087dd36deaf37af0ef35")
        return

    # Define function to load stock data
    def load_data(ticker, start, end):
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data

    # Proceed only if the user clicks a button
    if st.button("Run Causal Inference Model"):
        try:
            # Initialize FRED API
            fred = Fred(api_key=fred_api_key)

            # Load training and testing stock data
            train_data = load_data(stock_ticker, START, END)
            test_data = load_data(stock_ticker, END, TODAY.strftime("%Y-%m-%d"))

            if train_data.empty or test_data.empty:
                st.error("Failed to download stock data.")
                return

            st.subheader(f"Training Stock Data for {stock_ticker}")
            st.dataframe(train_data)

            st.subheader(f"Testing Stock Data for {stock_ticker}")
            st.dataframe(test_data)

            # Fetch CPI and Interest Rate data from FRED
            cpi_data = fred.get_series('CPIAUCSL', START, TODAY)  # CPI data
            interest_rate_data = fred.get_series('FEDFUNDS', START, TODAY)  # Federal Funds Rate

            # Convert FRED data to DataFrames and align with stock data
            cpi_df = cpi_data.reset_index().rename(columns={"index": "Date", 0: "CPI"})
            interest_rate_df = interest_rate_data.reset_index().rename(columns={"index": "Date", 0: "InterestRate"})

            # Reset index and ensure single-level columns in all DataFrames
            train_data = train_data.reset_index(drop=True)
            train_data.columns = train_data.columns.get_level_values(0)  # Flatten multi-level columns if any

            cpi_df = cpi_df.reset_index(drop=True)  # Reset index and flatten if necessary
            cpi_df.columns = ['Date', 'CPI']

            interest_rate_df = interest_rate_df.reset_index(drop=True)  # Reset index and flatten if necessary
            interest_rate_df.columns = ['Date', 'InterestRate']

            # Ensure Date columns are in the same datetime format (datetime64[ns])
            train_data['Date'] = pd.to_datetime(train_data['Date']).dt.tz_localize(None)
            cpi_df['Date'] = pd.to_datetime(cpi_df['Date']).dt.tz_localize(None)
            interest_rate_df['Date'] = pd.to_datetime(interest_rate_df['Date']).dt.tz_localize(None)

            # Merge CPI and Interest Rate data with training stock data
            train_data = pd.merge(train_data, cpi_df, on="Date", how="left")
            train_data = pd.merge(train_data, interest_rate_df, on="Date", how="left")

            # Handle any remaining NaN values by interpolating and forward/backward filling
            train_data['CPI'].interpolate(method='linear', inplace=True)
            train_data['InterestRate'].fillna(method='ffill', inplace=True)
            train_data['InterestRate'].fillna(method='bfill', inplace=True)

            # Define the lag period (e.g., 30 days)
            lag_days_cpi = 50
            lag_days_int = 16

            # Add lagged CPI and Interest Rate columns to the training data
            train_data['CPI_lag'] = train_data['CPI'].shift(lag_days_cpi)
            train_data['InterestRate_lag'] = train_data['InterestRate'].shift(lag_days_int)

            # Drop rows with NaN values introduced by the lag operation
            df_train = train_data[['Date', 'Close', 'CPI', 'InterestRate', 'CPI_lag', 'InterestRate_lag']].dropna().rename(columns={"Date": "ds", "Close": "y"})

            # MODIFICATION STARTS HERE
            # Ensure that 'ds' in df_train is datetime without timezone
            df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)

            # Ensure that 'y' is float64
            df_train['y'] = df_train['y'].astype(np.float64)

            # Ensure that regressor columns in df_train are float64
            regressor_columns = ['CPI', 'InterestRate', 'CPI_lag', 'InterestRate_lag']
            for col in regressor_columns:
                df_train[col] = df_train[col].astype(np.float64)
            # MODIFICATION ENDS HERE

            # Initialize and train a new Prophet model with lagged regressors
            m_with_lag = Prophet()
            m_with_lag.add_regressor('CPI')
            m_with_lag.add_regressor('InterestRate')
            m_with_lag.add_regressor('CPI_lag')
            m_with_lag.add_regressor('InterestRate_lag')
            m_with_lag.fit(df_train)

            # Prepare the future DataFrame for predictions, including lagged regressors
            period = (pd.to_datetime(TODAY) - pd.to_datetime(END)).days
            future_with_lag = m_with_lag.make_future_dataframe(periods=period)
            future_with_lag = future_with_lag.merge(cpi_df, left_on='ds', right_on='Date', how='left').drop(columns=['Date'])
            future_with_lag = future_with_lag.merge(interest_rate_df, left_on='ds', right_on='Date', how='left').drop(columns=['Date'])

            # Create lagged columns in the future data
            future_with_lag['CPI_lag'] = future_with_lag['CPI'].shift(lag_days_cpi)
            future_with_lag['InterestRate_lag'] = future_with_lag['InterestRate'].shift(lag_days_int)

            # Interpolate and forward/backward fill any missing values in the future data
            future_with_lag['CPI'].interpolate(method='linear', inplace=True)
            future_with_lag['InterestRate'].fillna(method='ffill', inplace=True)
            future_with_lag['InterestRate'].fillna(method='bfill', inplace=True)
            future_with_lag['CPI_lag'].interpolate(method='linear', inplace=True)
            future_with_lag['InterestRate_lag'].fillna(method='ffill', inplace=True)
            future_with_lag['InterestRate_lag'].fillna(method='bfill', inplace=True)

            # Ensure no missing values in the future_with_lag DataFrame before forecasting
            future_with_lag['CPI'].interpolate(method='linear', inplace=True)
            future_with_lag['CPI'].fillna(method='ffill', inplace=True)
            future_with_lag['CPI'].fillna(method='bfill', inplace=True)

            future_with_lag['InterestRate'].fillna(method='ffill', inplace=True)
            future_with_lag['InterestRate'].fillna(method='bfill', inplace=True)

            future_with_lag['CPI_lag'].interpolate(method='linear', inplace=True)
            future_with_lag['CPI_lag'].fillna(method='ffill', inplace=True)
            future_with_lag['CPI_lag'].fillna(method='bfill', inplace=True)

            future_with_lag['InterestRate_lag'].fillna(method='ffill', inplace=True)
            future_with_lag['InterestRate_lag'].fillna(method='bfill', inplace=True)

            # MODIFICATION STARTS HERE
            # Ensure that 'ds' in future_with_lag is datetime without timezone
            future_with_lag['ds'] = pd.to_datetime(future_with_lag['ds']).dt.tz_localize(None)

            # Ensure that regressor columns in future_with_lag are float64
            for col in regressor_columns:
                future_with_lag[col] = future_with_lag[col].astype(np.float64)
            # MODIFICATION ENDS HERE

            # Forecast with lagged regressors
            forecast_with_lag = m_with_lag.predict(future_with_lag)

            # Display forecast data with lagged regressors
            st.subheader('Causal Forecast data with Lagged Regressors:')
            st.write(forecast_with_lag.tail())

            # Plot forecast with lagged regressors
            st.subheader(f'Causal Forecast plot with Lagged Regressors from {START} to {TODAY}')
            fig1 = m_with_lag.plot(forecast_with_lag)
            st.pyplot(fig1)

            # Forecast components with lagged regressors
            fig2 = m_with_lag.plot_components(forecast_with_lag)
            st.pyplot(fig2)

            # Compare forecasted values with actual values
            forecast_filtered_lag = forecast_with_lag[forecast_with_lag['ds'] >= END][['ds', 'yhat']].reset_index(drop=True)
            test_data_filtered = test_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).reset_index(drop=True)

            # Ensure both DataFrames have single-level columns by resetting the index
            forecast_filtered_lag.columns = forecast_filtered_lag.columns.get_level_values(0)  # Flatten columns if multi-level
            test_data_filtered.columns = test_data_filtered.columns.get_level_values(0)  # Flatten columns if multi-level

            # Remove timezone information from the 'ds' columns in both DataFrames
            forecast_filtered_lag['ds'] = forecast_filtered_lag['ds'].dt.tz_localize(None)
            test_data_filtered['ds'] = test_data_filtered['ds'].dt.tz_localize(None)

            # Merge forecasted and actual values on date
            comparison_df_lag = pd.merge(forecast_filtered_lag, test_data_filtered, on='ds', how='inner')
            comparison_df_lag = comparison_df_lag.rename(columns={"yhat": "Forecasted", "y": "Actual"})

            # Calculate accuracy metrics for the lagged causal model
            mae_causal_lag = mean_absolute_error(comparison_df_lag['Actual'], comparison_df_lag['Forecasted'])
            mse_causal_lag = mean_squared_error(comparison_df_lag['Actual'], comparison_df_lag['Forecasted'])
            rmse_causal_lag = np.sqrt(mse_causal_lag)
            mape_causal_lag = np.mean(np.abs((comparison_df_lag['Actual'] - comparison_df_lag['Forecasted']) / comparison_df_lag['Actual'])) * 100

            # Display comparison and accuracy metrics for lagged causal model
            st.subheader("Forecast vs Actual with Causal Model (Lagged):")
            st.write(comparison_df_lag)

            st.write(f"Causal Model with Lagged Regressors Mean Absolute Error (MAE): {mae_causal_lag:.2f}")
            st.write(f"Causal Model with Lagged Regressors Mean Squared Error (MSE): {mse_causal_lag:.2f}")
            st.write(f"Causal Model with Lagged Regressors Root Mean Squared Error (RMSE): {rmse_causal_lag:.2f}")
            st.write(f"Causal Model with Lagged Regressors Mean Absolute Percentage Error (MAPE): {mape_causal_lag:.2f}%")

            # Proceed to include RSI and SMA if the user wants
            include_rsi = st.checkbox("Include RSI in the model?", value=True)
            include_sma = st.checkbox("Include SMA in the model?", value=True)

            if include_rsi or include_sma:
                # Calculate RSI and SMA for training data
                if include_rsi:
                    train_data = calculate_rsi(train_data, window=14)
                    train_data['RSI'].fillna(method='ffill', inplace=True)
                    train_data['RSI'].fillna(method='bfill', inplace=True)
                    df_train['RSI'] = train_data['RSI']

                if include_sma:
                    train_data = calculate_sma(train_data, window=30)
                    train_data['SMA'].fillna(method='ffill', inplace=True)
                    train_data['SMA'].fillna(method='bfill', inplace=True)
                    df_train['SMA'] = train_data['SMA']

                # Ensure regressor columns are float64
                additional_regressors = []
                if include_rsi:
                    df_train['RSI'] = df_train['RSI'].astype(np.float64)
                    additional_regressors.append('RSI')
                if include_sma:
                    df_train['SMA'] = df_train['SMA'].astype(np.float64)
                    additional_regressors.append('SMA')

                # Initialize and train a new Prophet model with additional regressors
                m_with_lag_rsi_sma = Prophet()
                m_with_lag_rsi_sma.add_regressor('CPI')
                m_with_lag_rsi_sma.add_regressor('InterestRate')
                m_with_lag_rsi_sma.add_regressor('CPI_lag')
                m_with_lag_rsi_sma.add_regressor('InterestRate_lag')
                if include_rsi:
                    m_with_lag_rsi_sma.add_regressor('RSI')
                if include_sma:
                    m_with_lag_rsi_sma.add_regressor('SMA')
                m_with_lag_rsi_sma.fit(df_train)

                # Prepare future DataFrame
                future_with_lag_rsi_sma = future_with_lag.copy()
                if include_rsi:
                    last_rsi = train_data['RSI'].iloc[-1]
                    future_with_lag_rsi_sma['RSI'] = last_rsi
                    future_with_lag_rsi_sma['RSI'].fillna(method='ffill', inplace=True)
                    future_with_lag_rsi_sma['RSI'].fillna(method='bfill', inplace=True)
                    future_with_lag_rsi_sma['RSI'] = future_with_lag_rsi_sma['RSI'].astype(np.float64)
                if include_sma:
                    last_sma = train_data['SMA'].iloc[-1]
                    future_with_lag_rsi_sma['SMA'] = last_sma
                    future_with_lag_rsi_sma['SMA'].fillna(method='ffill', inplace=True)
                    future_with_lag_rsi_sma['SMA'].fillna(method='bfill', inplace=True)
                    future_with_lag_rsi_sma['SMA'] = future_with_lag_rsi_sma['SMA'].astype(np.float64)

                # Forecast
                forecast_with_lag_rsi_sma = m_with_lag_rsi_sma.predict(future_with_lag_rsi_sma)

                # Display forecast data with RSI and SMA included
                st.subheader('Causal Forecast data with Lagged Regressors, RSI, and SMA:')
                st.write(forecast_with_lag_rsi_sma.tail())

                # Plot forecast with RSI and SMA included
                st.subheader(f'Causal Forecast plot with Lagged Regressors, RSI, and SMA from {START} to {TODAY}')
                fig3 = m_with_lag_rsi_sma.plot(forecast_with_lag_rsi_sma)
                st.pyplot(fig3)

                # Forecast components with RSI and SMA included
                fig4 = m_with_lag_rsi_sma.plot_components(forecast_with_lag_rsi_sma)
                st.pyplot(fig4)

                # Compare forecasted values with actual values
                forecast_filtered_lag_rsi_sma = forecast_with_lag_rsi_sma[forecast_with_lag_rsi_sma['ds'] >= END][['ds', 'yhat']].reset_index(drop=True)

                # Ensure the 'ds' columns have matching datetime format and timezone info removed
                forecast_filtered_lag_rsi_sma['ds'] = forecast_filtered_lag_rsi_sma['ds'].dt.tz_localize(None)
                test_data_filtered['ds'] = test_data_filtered['ds'].dt.tz_localize(None)

                # Merge forecasted and actual values
                comparison_df_lag_rsi_sma = pd.merge(forecast_filtered_lag_rsi_sma, test_data_filtered, on='ds', how='inner')
                comparison_df_lag_rsi_sma = comparison_df_lag_rsi_sma.rename(columns={"yhat": "Forecasted", "y": "Actual"})

                # Calculate accuracy metrics
                mae_causal_lag_rsi_sma = mean_absolute_error(comparison_df_lag_rsi_sma['Actual'], comparison_df_lag_rsi_sma['Forecasted'])
                mse_causal_lag_rsi_sma = mean_squared_error(comparison_df_lag_rsi_sma['Actual'], comparison_df_lag_rsi_sma['Forecasted'])
                rmse_causal_lag_rsi_sma = np.sqrt(mse_causal_lag_rsi_sma)
                mape_causal_lag_rsi_sma = np.mean(np.abs((comparison_df_lag_rsi_sma['Actual'] - comparison_df_lag_rsi_sma['Forecasted']) / comparison_df_lag_rsi_sma['Actual'])) * 100

                # Display comparison and accuracy metrics
                st.subheader("Forecast vs Actual with Causal Model (Lagged + RSI + SMA):")
                st.write(comparison_df_lag_rsi_sma)

                st.write(f"Causal Model with Lagged Regressors, RSI, and SMA Mean Absolute Error (MAE): {mae_causal_lag_rsi_sma:.2f}")
                st.write(f"Causal Model with Lagged Regressors, RSI, and SMA Mean Squared Error (MSE): {mse_causal_lag_rsi_sma:.2f}")
                st.write(f"Causal Model with Lagged Regressors, RSI, and SMA Root Mean Squared Error (RMSE): {rmse_causal_lag_rsi_sma:.2f}")
                st.write(f"Causal Model with Lagged Regressors, RSI, and SMA Mean Absolute Percentage Error (MAPE): {mape_causal_lag_rsi_sma:.2f}%")

                # Perform cross-validation
                st.subheader("Cross-Validated Performance Metrics for Model with Lagged Regressors, RSI, and SMA:")
                # Set horizon and initial period according to your needs (e.g., 180 days for the horizon, 365 days for initial)
                df_cv = cross_validation(
                    m_with_lag_rsi_sma,  # Model with RSI and SMA
                    initial='1460 days',  # Initial training period
                    period='90 days',   # Period between cutoffs
                    horizon='180 days'   # Forecast horizon
                )

                # Calculate performance metrics
                df_p = performance_metrics(df_cv)

                # Display the performance metrics to understand cross-validated accuracy
                st.write(df_p)

                st.write(f"Cross-Validated Mean Absolute Error (MAE): {df_p['mae'].mean():.2f}")
                st.write(f"Cross-Validated Mean Squared Error (MSE): {df_p['mse'].mean():.2f}")
                st.write(f"Cross-Validated Root Mean Squared Error (RMSE): {df_p['rmse'].mean():.2f}")
                st.write(f"Cross-Validated Mean Absolute Percentage Error (MAPE): {df_p['mape'].mean() * 100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

# The rest of the code remains the same
# Prediction Function
def prediction():
    st.title("Stock Prediction")

    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Convert float columns to float64
    float_columns = df_train.select_dtypes(include=['float']).columns
    for col in float_columns:
        df_train[col] = df_train[col].astype(np.float64)

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    # Trend Analysis
    historical_trend = identify_trend(data['Close'])
    forecast_trend = identify_trend(forecast['yhat'])
    
    st.subheader("Trend Analysis")
    st.write(f"Historical Trend: {historical_trend}")
    st.write(f"Forecast Trend: {forecast_trend}")

# Risk Mitigation Function
def risk_mitigation():
    st.title("Risk Mitigation")

    stock_ticker = st.text_input("Enter the stock ticker:", value="MSFT")
    if not stock_ticker:
        st.error("Please enter a stock ticker.")
        return

    df = download_stock_data(stock_ticker)
    if df.empty:
        st.error("Failed to download stock data.")
        return

    st.subheader(f"Stock Data for {stock_ticker}")
    st.dataframe(df)

    returns = df['Adj Close'].pct_change().dropna()

    # Function to create a color-coded metric
    def color_metric(label, value, interpretation):
        if interpretation == "Good":
            color = "green"
        elif interpretation == "Moderate":
            color = "orange"
        else:
            color = "red"
        st.markdown(f"<h3 style='color: {color};'>{label}: {value}</h3>", unsafe_allow_html=True)

    # Volatility Analysis
    volatility = calculate_volatility(returns)
    st.subheader("Volatility Analysis")
    if volatility < 0.15:
        interpretation = "Low"
    elif volatility < 0.30:
        interpretation = "Moderate"
    else:
        interpretation = "High"
    color_metric("Annualized Volatility", f"{volatility:.2%}", interpretation)

    with st.expander("Learn more about Volatility"):
        st.write("""Volatility measures the degree of variation in a stock's price over time. 
        Higher volatility indicates higher risk but also potential for higher returns.""")

    # Risk-Adjusted Returns
    sharpe_ratio = calculate_sharpe_ratio(returns)
    st.subheader("Risk-Adjusted Returns")
    if sharpe_ratio < 0.5:
        interpretation = "Poor"
    elif sharpe_ratio < 1.0:
        interpretation = "Below Average"
    elif sharpe_ratio < 2.0:
        interpretation = "Good"
    else:
        interpretation = "Very Good"
    color_metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", interpretation)

    with st.expander("Learn more about Sharpe Ratio"):
        st.write("""The Sharpe Ratio measures the performance of an investment compared to a risk-free asset, 
        after adjusting for its risk.""")

    # Value at Risk (VaR)
    var_95 = calculate_var(returns)
    st.subheader("Value at Risk (VaR)")
    if abs(var_95) < 0.02:
        interpretation = "Low Risk"
    elif abs(var_95) < 0.05:
        interpretation = "Moderate Risk"
    else:
        interpretation = "High Risk"
    color_metric("95% VaR", f"{var_95:.2%}", interpretation)

    with st.expander("Learn more about Value at Risk (VaR)"):
        st.write(f"With 95% confidence, we do not expect the stock to lose more than {abs(var_95):.2%} of its value in a single day.")

    # Stress Testing
    st.subheader("Stress Testing")

    st.write("Stress testing simulates extreme market conditions to assess the potential impact on portfolio performance.")

    market_crash_percent = st.slider("Simulate market crash by (percent drop):", min_value=5, max_value=50, value=20)

    # Adjust the prices based on the percentage drop
    stressed_prices = df['Adj Close'] * (1 - market_crash_percent / 100)
    stressed_returns = stressed_prices.pct_change().dropna()

    # Recalculate metrics based on the stressed returns
    stressed_volatility = calculate_volatility(stressed_returns)
    stressed_sharpe_ratio = calculate_sharpe_ratio(stressed_returns)
    stressed_var_95 = calculate_var(stressed_returns)

    # Display the recalculated values
    st.write(f"Simulated Volatility after {market_crash_percent}% drop: {stressed_volatility:.2%}")
    st.write(f"Simulated Sharpe Ratio after {market_crash_percent}% drop: {stressed_sharpe_ratio:.2f}")
    st.write(f"Simulated 95% VaR after {market_crash_percent}% drop: {stressed_var_95:.2%}")

    # Plot the original and stressed prices
    fig_stress, ax_stress = plt.subplots()
    ax_stress.plot(df['Adj Close'], label="Original Prices")
    ax_stress.plot(stressed_prices, label=f"Prices after {market_crash_percent}% drop", linestyle='--')
    ax_stress.set_title("Original vs. Stressed Prices")
    ax_stress.legend()
    st.pyplot(fig_stress)

    # Drawdown Analysis
    st.subheader("Drawdown Analysis")
    st.write("Drawdown refers to the peak-to-trough decline during a specific period. It helps understand the risk of a significant loss in the stock's value.")
    
    drawdown, max_drawdown = calculate_drawdowns(df['Adj Close'])
    st.write(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Plot drawdowns
    fig_drawdown, ax_drawdown = plt.subplots()
    ax_drawdown.plot(df.index, drawdown, label='Drawdown', color='red')
    ax_drawdown.set_title('Drawdown Over Time')
    ax_drawdown.set_ylabel('Drawdown')
    ax_drawdown.legend()
    st.pyplot(fig_drawdown)

    # Stop-Loss and Take-Profit Recommendations
    st.subheader("Stop-Loss and Take-Profit Recommendations")

    stop_loss_level = df['Adj Close'].iloc[-1] - (volatility * df['Adj Close'].iloc[-1])
    take_profit_level = df['Adj Close'].iloc[-1] + (volatility * df['Adj Close'].iloc[-1])

    st.write(f"Recommended Stop-Loss Level: ${stop_loss_level:.2f}")
    st.write(f"Recommended Take-Profit Level: ${take_profit_level:.2f}")

    # Scenario Analysis
    st.subheader("Scenario Analysis")

    st.write("Simulate different market conditions to evaluate how your stock might perform.")

    # Select Scenario first
    scenario = st.selectbox("Select Scenario", ["Economic Downturn", "Interest Rate Hike", "Inflation Rise", "Bull Market", "Multiple Simulations"])

    # Custom input for percentage changes for scenarios
    economic_downturn = st.number_input("Economic Downturn (e.g., -25 for 25% drop)", value=-25.0)
    interest_rate_hike = st.number_input("Interest Rate Hike (e.g., -10 for 10% drop)", value=-10.0)
    inflation_rise = st.number_input("Inflation Rise (e.g., -5 for 5% drop)", value=-5.0)
    bull_market = st.number_input("Bull Market (e.g., 20 for 20% increase)", value=20.0)

    adjustment = 0

    if scenario != "Multiple Simulations":
        if scenario == "Economic Downturn":
            adjustment = economic_downturn / 100  # Convert percentage to decimal
        elif scenario == "Interest Rate Hike":
            adjustment = interest_rate_hike / 100
        elif scenario == "Inflation Rise":
            adjustment = inflation_rise / 100
        elif scenario == "Bull Market":
            adjustment = bull_market / 100

        scenario_prices = df['Adj Close'] * (1 + adjustment)

        st.write(f"Simulated performance in {scenario} with {adjustment*100:.0f}% adjustment.")

        fig_scenario, ax_scenario = plt.subplots()
        ax_scenario.plot(df['Adj Close'], label="Original Prices")
        ax_scenario.plot(scenario_prices, label=f"Prices after {scenario} adjustment", linestyle='--')
        ax_scenario.set_title(f"Original vs. {scenario} Adjusted Prices")
        ax_scenario.legend()
        st.pyplot(fig_scenario)

    # Multiple Simulations
    else:
        num_simulations = st.number_input("Enter number of simulations", min_value=2, max_value=100, value=5)
        adjustment_list = []

        for i in range(num_simulations):
            scenario = st.selectbox(f"Select Scenario for Simulation {i+1}", ["Economic Downturn", "Interest Rate Hike", "Inflation Rise", "Bull Market"], key=i)

            if scenario == "Economic Downturn":
                adjustment = economic_downturn / 100
            elif scenario == "Interest Rate Hike":
                adjustment = interest_rate_hike / 100
            elif scenario == "Inflation Rise":
                adjustment = inflation_rise / 100
            elif scenario == "Bull Market":
                adjustment = bull_market / 100

            adjustment_list.append(adjustment)

        fig_multi_scenario, ax_multi_scenario = plt.subplots()
        ax_multi_scenario.plot(df['Adj Close'], label="Original Prices")

        for i, adj in enumerate(adjustment_list):
            multi_scenario_prices = df['Adj Close'] * (1 + adj)
            ax_multi_scenario.plot(multi_scenario_prices, linestyle='--', label=f"Simulation {i+1}")

        ax_multi_scenario.set_title("Original vs. Multiple Simulations Adjusted Prices")
        ax_multi_scenario.legend()
        st.pyplot(fig_multi_scenario)

    # Portfolio Optimization Section
    st.subheader("Portfolio Optimization")

    st.write("""
    Portfolio optimization is the process of selecting the best portfolio (asset distribution) 
    according to some objective. Typically, an investor wishes to maximize expected return while minimizing risk, 
    or maximize return given a certain risk tolerance.
    """)

    stock_symbols = st.multiselect("Select stocks for portfolio optimization", ['MSFT', 'AAPL', 'GOOG', 'AMZN'], default=['MSFT', 'AAPL'])

    if len(stock_symbols) > 1:
        stock_data = yf.download(stock_symbols, start="2020-01-01")['Adj Close']

        optimal_weights, optimized_return, optimized_volatility, optimized_sharpe_ratio = portfolio_optimization(stock_data, stock_symbols)

        st.write("### Optimal Portfolio Allocation")
        for symbol, weight in zip(stock_symbols, optimal_weights):
            st.write(f"{symbol}: {weight:.2%}")

        st.write(f"Expected Annualized Return: {optimized_return:.2%}")
        st.write(f"Expected Annualized Volatility: {optimized_volatility:.2%}")
        st.write(f"Expected Sharpe Ratio: {optimized_sharpe_ratio:.2f}")

        # Plotting the portfolio's stock price evolution
        fig, ax = plt.subplots()
        stock_data.plot(ax=ax, title="Stock Price Evolution of Selected Portfolio", legend=True)
        st.pyplot(fig)
    else:
        st.error("Please select at least two stocks for portfolio optimization.")

# Main Function
def main():
    page = st.sidebar.selectbox("Select Page", ["Monte Carlo Simulation", "Causal Inference", "Prediction", "Risk Mitigation"])

    if page == "Monte Carlo Simulation":
        monte_carlo_simulation()
    elif page == "Causal Inference":
        causal_inference()
    elif page == "Prediction":
        prediction()
    elif page == "Risk Mitigation":
        risk_mitigation()

if _name_ == "_main_":
    main()