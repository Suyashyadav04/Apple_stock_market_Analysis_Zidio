import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Apple Stock Analysis & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Apple Stock Analysis & Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Add option to test with sample data
use_sample_data = st.sidebar.checkbox("Use Sample Data (for testing)")

if not use_sample_data:
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2024, 12, 31))
else:
    start_date = datetime(2020, 1, 1).date()
    end_date = datetime(2024, 12, 31).date()
    st.sidebar.info("Using sample data mode")

forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30)

# Add manual data entry option
st.sidebar.markdown("---")
st.sidebar.subheader("Alternative: Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def create_sample_data():
    """Create sample data for testing when yfinance fails"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    dates = dates[dates.weekday < 5]
    
    np.random.seed(42)
    n_days = len(dates)
    
    base_price = 150
    price_changes = np.random.normal(0.001, 0.02, n_days)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))
    
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    volumes = np.random.lognormal(15, 0.5, n_days).astype(int)
    
    df_sample = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })
    
    return df_sample

@st.cache_data
def load_stock_data(start_date, end_date):
    """Load and clean Apple stock data"""
    try:
        with st.spinner('Downloading Apple stock data...'):
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            data = yf.download(
                'AAPL', 
                start=start_str, 
                end=end_str, 
                progress=False,
                auto_adjust=True
            )
            
        if data.empty:
            st.error("No data found for the selected date range")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == 'AAPL' else '_'.join(col).strip() for col in data.columns.values]
        
        data = data.reset_index()
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = data.columns.tolist()
        
        column_mapping = {}
        for col in required_cols:
            if col in available_cols:
                column_mapping[col] = col
            elif col.lower() in [c.lower() for c in available_cols]:
                actual_col = next(c for c in available_cols if c.lower() == col.lower())
                column_mapping[col] = actual_col
            else:
                st.error(f"Required column '{col}' not found in data")
                return None
        
        df = data[[column_mapping[col] for col in required_cols]].copy()
        df.columns = required_cols
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        df = df.dropna()
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) == 0:
            st.error("No valid data after cleaning")
            return None
            
        st.success(f"Successfully loaded {len(df)} records from {df['Date'].min().date()} to {df['Date'].max().date()}")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data with error handling
df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in df.columns for col in required_cols):
            st.sidebar.success("CSV file loaded successfully!")
        else:
            st.sidebar.error(f"CSV must contain columns: {required_cols}")
            df = None
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {str(e)}")
        df = None

elif use_sample_data:
    df = create_sample_data()
    st.info("📊 Using sample data for demonstration")

else:
    try:
        df = load_stock_data(start_date, end_date)
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.info("💡 Try enabling 'Use Sample Data' in the sidebar to test the app with sample data")
        df = None

if df is not None:
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{len(df)} days")
    with col3:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    with col4:
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
        st.metric("Total Change", f"${price_change:.2f}")
    
    with st.expander("View Raw Data"):
        st.dataframe(df.head(10))
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="apple_stock_data.csv",
            mime="text/csv"
        )
    
    # Visualizations
    st.header("📈 Stock Price Visualization")
    
    fig_basic = go.Figure()
    fig_basic.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    fig_basic.update_layout(
        title="Apple Stock Closing Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_basic, use_container_width=True)
    
    # Moving averages
    st.subheader("Moving Averages")
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='50-day MA', line=dict(color='orange')))
    fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], name='200-day MA', line=dict(color='red')))
    
    fig_ma.update_layout(
        title="Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Volume chart
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'))
    fig_volume.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=False
    )
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Forecasting section
    st.header("🔮 Stock Price Forecasting")
    
    df_forecast = df.set_index('Date').copy()
    
    forecast_results = {}
    
    # ARIMA Forecasting
    st.subheader("ARIMA Model")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        with st.spinner('Training ARIMA model...'):
            model_arima = ARIMA(df_forecast['Close'], order=(5, 1, 2))
            model_fit_arima = model_arima.fit()
            
            last_date = df_forecast.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=forecast_days, freq='D')
            
            forecast_arima = model_fit_arima.forecast(steps=forecast_days)
            
            forecast_results['ARIMA'] = {
                'dates': forecast_dates,
                'values': forecast_arima,
                'model': model_fit_arima
            }
            
            fig_arima = go.Figure()
            
            fig_arima.add_trace(go.Scatter(
                x=df_forecast.index[-100:],
                y=df_forecast['Close'][-100:],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            fig_arima.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_arima,
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig_arima.update_layout(
                title="ARIMA Forecast - Apple Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_arima, use_container_width=True)
            
            with st.expander("ARIMA Model Summary"):
                st.text(str(model_fit_arima.summary()))
        
        st.success("ARIMA model completed successfully!")
        
    except ImportError:
        st.warning("ARIMA requires statsmodels. Install with: pip install statsmodels")
    except Exception as e:
        st.error(f"ARIMA model error: {str(e)}")
    
    # Prophet Forecasting
    st.subheader("Prophet Model")
    try:
        from prophet import Prophet
        
        with st.spinner('Training Prophet model...'):
            prophet_df = df_forecast.reset_index()[['Date', 'Close']].copy()
            prophet_df.columns = ['ds', 'y']
            
            model_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model_prophet.fit(prophet_df)
            
            future = model_prophet.make_future_dataframe(periods=forecast_days)
            forecast_prophet = model_prophet.predict(future)
            
            forecast_results['Prophet'] = {
                'dates': forecast_prophet['ds'][-forecast_days:],
                'values': forecast_prophet['yhat'][-forecast_days:],
                'model': model_prophet
            }
            
            fig_prophet = model_prophet.plot(forecast_prophet)
            st.pyplot(fig_prophet)
            
            fig_components = model_prophet.plot_components(forecast_prophet)
            st.pyplot(fig_components)
        
        st.success("Prophet model completed successfully!")
        
    except ImportError:
        st.warning("Prophet requires fbprophet. Install with: pip install prophet")
    except Exception as e:
        st.error(f"Prophet model error: {str(e)}")
    
    # LSTM Forecasting
    st.subheader("LSTM Model")
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        import tensorflow as tf
        
        tf.get_logger().setLevel('ERROR')
        
        min_data_points = 100
        if len(df_forecast) < min_data_points:
            st.warning(f"LSTM requires at least {min_data_points} data points. Current: {len(df_forecast)}. Skipping.")
        else:
            with st.spinner('Training LSTM model...'):
                close_data = df_forecast[['Close']].copy()
                close_data = close_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(close_data) < min_data_points:
                    st.error(f"After cleaning, insufficient data. Need {min_data_points}, have {len(close_data)}")
                else:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(close_data.values)
                    
                    if len(scaled_data) == 0:
                        st.error("Data scaling failed")
                    else:
                        train_size = int(len(scaled_data) * 0.8)
                        train_data = scaled_data[:train_size]
                        
                        sequence_length = min(60, len(train_data) - 10)
                        
                        if sequence_length < 10:
                            st.error(f"Insufficient training data")
                        else:
                            x_train, y_train = [], []
                            
                            for i in range(sequence_length, len(train_data)):
                                x_train.append(train_data[i-sequence_length:i])
                                y_train.append(train_data[i])
                            
                            x_train = np.array(x_train)
                            y_train = np.array(y_train)
                            
                            if len(x_train) == 0:
                                st.error("Failed to create training sequences")
                            else:
                                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                                
                                st.info(f"Training LSTM with {len(x_train)} samples")
                                
                                model_lstm = Sequential()
                                model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                                model_lstm.add(LSTM(units=50))
                                model_lstm.add(Dense(1))
                                
                                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                                history = model_lstm.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
                                
                                if len(scaled_data) >= sequence_length:
                                    last_sequence = scaled_data[-sequence_length:].copy()
                                    lstm_forecast = []
                                    
                                    for _ in range(forecast_days):
                                        next_pred = model_lstm.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                                        lstm_forecast.append(next_pred[0, 0])
                                        last_sequence = np.append(last_sequence[1:], next_pred)
                                    
                                    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
                                    
                                    last_date = df_forecast.index[-1]
                                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
                                    
                                    forecast_results['LSTM'] = {
                                        'dates': forecast_dates,
                                        'values': lstm_forecast,
                                        'model': model_lstm
                                    }
                                    
                                    fig_lstm = go.Figure()
                                    plot_history = min(100, len(df_forecast))
                                    
                                    fig_lstm.add_trace(go.Scatter(
                                        x=df_forecast.index[-plot_history:],
                                        y=df_forecast['Close'][-plot_history:],
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='blue')
                                    ))
                                    
                                    fig_lstm.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=lstm_forecast,
                                        mode='lines',
                                        name='LSTM Forecast',
                                        line=dict(color='green', dash='dash')
                                    ))
                                    
                                    fig_lstm.update_layout(
                                        title="LSTM Forecast - Apple Stock Price",
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_lstm, use_container_width=True)
                                    
                                    fig_loss = go.Figure()
                                    fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                                    fig_loss.update_layout(title="LSTM Training Loss", yaxis_title="Loss")
                                    st.plotly_chart(fig_loss, use_container_width=True)
                                    
                                    st.success("✅ LSTM model completed successfully!")
                                else:
                                    st.error("Insufficient data for forecasting")
        
    except ImportError as ie:
        st.warning(f"LSTM requires TensorFlow and scikit-learn. Install with: pip install tensorflow scikit-learn")
    except Exception as e:
        st.error(f"LSTM model error: {str(e)}")
    
    # Model Comparison
    if len(forecast_results) > 1:
        st.header("🏆 Model Comparison")
        
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Scatter(
            x=df_forecast.index[-100:],
            y=df_forecast['Close'][-100:],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=3)
        ))
        
        colors = ['red', 'orange', 'green', 'purple']
        for i, (model_name, results) in enumerate(forecast_results.items()):
            fig_combined.add_trace(go.Scatter(
                x=results['dates'],
                y=results['values'],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=colors[i % len(colors)], dash='dash', width=2)
            ))
        
        fig_combined.update_layout(
            title="Forecast Comparison - All Models",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=600
        )
        st.plotly_chart(fig_combined, use_container_width=True)
        
        st.subheader("Forecast Summary")
        
        summary_data = []
        for model_name, results in forecast_results.items():
            forecast_values = results['values']
            
            if isinstance(forecast_values, pd.Series):
                forecast_array = forecast_values.values
            else:
                forecast_array = np.array(forecast_values)
            
            if len(forecast_array) > 0:
                try:
                    next_day = f"${forecast_array[0]:.2f}"
                    next_week = f"${forecast_array[6]:.2f}" if len(forecast_array) > 6 else "N/A"
                    final_day = f"${forecast_array[-1]:.2f}"
                    avg_forecast = f"${np.mean(forecast_array):.2f}"
                    forecast_range = f"${np.min(forecast_array):.2f} - ${np.max(forecast_array):.2f}"
                    
                    summary_data.append({
                        'Model': model_name,
                        'Next Day': next_day,
                        'Next Week': next_week,
                        'Final Day': final_day,
                        'Avg Forecast': avg_forecast,
                        'Forecast Range': forecast_range
                    })
                except Exception as e:
                    st.warning(f"Error processing {model_name} forecast: {str(e)}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
    
    # Additional Analysis
    st.header("📊 Additional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Statistics")
        stats_data = {
            'Metric': ['Current Price', 'All-time High', 'All-time Low', 'Average Price', 'Volatility (Std Dev)'],
            'Value': [
                f"${df['Close'].iloc[-1]:.2f}",
                f"${df['Close'].max():.2f}",
                f"${df['Close'].min():.2f}",
                f"${df['Close'].mean():.2f}",
                f"${df['Close'].std():.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    with col2:
        st.subheader("Recent Performance")
        recent_data = df.tail(5)[['Date', 'Close', 'Volume']].copy()
        recent_data['Daily Change'] = recent_data['Close'].pct_change() * 100
        recent_data['Daily Change'] = recent_data['Daily Change'].round(2)
        st.dataframe(recent_data, use_container_width=True)
    
    st.subheader("Price-Volume Correlation")
    correlation = df['Close'].corr(df['Volume'])
    st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    fig_scatter = px.scatter(df, x='Volume', y='Close', title='Price vs Volume Correlation')
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.error("Unable to load stock data. Please check your settings and try again.")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit | Data provided by Yahoo Finance | "
    "Project by Suyash Yadav"
)