import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
import datetime as dt
from datetime import timedelta
import folium
from streamlit_folium import st_folium
import requests
import time

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Market Predictor | India",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme with neon green accents
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00ff88 0%, #00cc6a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #00ff88;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff88;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        color: #aaa;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 2px solid #00ff88;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
        text-align: center;
    }
    
    .prediction-title {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .prediction-item {
        margin: 0.8rem 0;
        font-size: 1.1rem;
    }
    
    .prediction-value {
        color: #00ff88;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        color: #888;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #000;
        border: 1px solid #00ff88;
        font-weight: 700;
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #1a1a1a;
        border: 1px solid #333;
        color: #fff;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #000;
        border: none;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
        transform: translateY(-2px);
    }
    
    .status-positive {
        color: #00ff88;
    }
    
    .status-negative {
        color: #ff4757;
    }
    
    .footer {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-title {
        color: #00ff88;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .disclaimer {
        color: #888;
        font-size: 0.9rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        
    def fetch_stock_data(self, symbol, period="2y"):
        """Fetch stock data using yfinance"""
        try:
            # Add .NS for NSE stocks if not already present
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = symbol + '.NS'
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(data[['Close']].values)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def predict_stock_price(self, data, days_ahead=30):
        """Generate stock price predictions"""
        # Prepare data
        X, y = self.prepare_lstm_data(data)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build and train model
        self.model = self.build_lstm_model((X_train.shape[1], 1))
        
        with st.spinner("Training AI model..."):
            self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Generate predictions
        last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)
        predictions = []
        
        for _ in range(days_ahead):
            pred = self.model.predict(last_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Calculate confidence intervals
        volatility = data['Close'].pct_change().std()
        confidence_bands = []
        
        for i, pred in enumerate(predictions):
            uncertainty = volatility * np.sqrt(i + 1) * pred[0]
            confidence_bands.append({
                'prediction': pred[0],
                'upper': pred[0] + 1.96 * uncertainty,
                'lower': pred[0] - 1.96 * uncertainty
            })
        
        return confidence_bands

class RealEstatePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def get_city_areas(self):
        """Get popular areas for each city"""
        return {
            'Mumbai': ['Bandra West', 'Andheri East', 'Powai', 'Lower Parel', 'Worli', 'Malad West', 'Goregaon East', 'Thane West'],
            'Delhi': ['Gurgaon', 'Noida', 'Dwarka', 'Rohini', 'Lajpat Nagar', 'Saket', 'Vasant Kunj', 'Greater Kailash'],
            'Bangalore': ['Whitefield', 'Koramangala', 'Indiranagar', 'Electronic City', 'HSR Layout', 'Marathahalli', 'JP Nagar', 'Banashankari'],
            'Hyderabad': ['Gachibowli', 'Madhapur', 'Kondapur', 'Kukatpally', 'Begumpet', 'Secunderabad', 'Miyapur', 'Manikonda'],
            'Chennai': ['OMR', 'Anna Nagar', 'T Nagar', 'Velachery', 'Adyar', 'Porur', 'Tambaram', 'Chrompet'],
            'Pune': ['Hinjewadi', 'Wakad', 'Aundh', 'Kothrud', 'Viman Nagar', 'Hadapsar', 'Magarpatta', 'Baner'],
            'Kolkata': ['Salt Lake', 'New Town', 'Ballygunge', 'Park Street', 'Garia', 'Howrah', 'Behala', 'Rajarhat'],
            'Ahmedabad': ['SG Highway', 'Satellite', 'Vastrapur', 'Prahlad Nagar', 'Bopal', 'Gota', 'Thaltej', 'Nikol']
        }
    
    def get_area_multiplier(self, city, area):
        """Get price multiplier based on area within city"""
        area_multipliers = {
            'Mumbai': {
                'Bandra West': 1.5, 'Lower Parel': 1.4, 'Worli': 1.6, 'Powai': 1.2,
                'Andheri East': 1.1, 'Malad West': 0.9, 'Goregaon East': 1.0, 'Thane West': 0.8
            },
            'Delhi': {
                'Gurgaon': 1.3, 'Greater Kailash': 1.4, 'Saket': 1.3, 'Vasant Kunj': 1.2,
                'Noida': 1.0, 'Dwarka': 0.9, 'Rohini': 0.8, 'Lajpat Nagar': 1.1
            },
            'Bangalore': {
                'Koramangala': 1.4, 'Indiranagar': 1.3, 'Whitefield': 1.1, 'HSR Layout': 1.2,
                'Electronic City': 0.9, 'Marathahalli': 1.0, 'JP Nagar': 1.0, 'Banashankari': 0.8
            },
            'Hyderabad': {
                'Gachibowli': 1.3, 'Madhapur': 1.2, 'Kondapur': 1.1, 'Begumpet': 1.2,
                'Kukatpally': 0.9, 'Secunderabad': 1.0, 'Miyapur': 0.8, 'Manikonda': 1.0
            },
            'Chennai': {
                'OMR': 1.2, 'Anna Nagar': 1.3, 'T Nagar': 1.4, 'Adyar': 1.3,
                'Velachery': 1.0, 'Porur': 0.9, 'Tambaram': 0.8, 'Chrompet': 0.7
            },
            'Pune': {
                'Hinjewadi': 1.1, 'Aundh': 1.2, 'Kothrud': 1.1, 'Viman Nagar': 1.0,
                'Wakad': 1.0, 'Hadapsar': 0.9, 'Magarpatta': 1.1, 'Baner': 1.2
            },
            'Kolkata': {
                'Salt Lake': 1.1, 'New Town': 1.0, 'Ballygunge': 1.3, 'Park Street': 1.4,
                'Garia': 0.8, 'Howrah': 0.7, 'Behala': 0.8, 'Rajarhat': 0.9
            },
            'Ahmedabad': {
                'SG Highway': 1.2, 'Satellite': 1.1, 'Vastrapur': 1.1, 'Prahlad Nagar': 1.0,
                'Bopal': 0.9, 'Gota': 0.9, 'Thaltej': 1.0, 'Nikol': 0.8
            }
        }
        return area_multipliers.get(city, {}).get(area, 1.0)
        
    def generate_sample_data(self, location, area, num_properties=100):
        """Generate sample real estate data for Indian cities with specific areas"""
        np.random.seed(42)
        
        # Indian city coordinates and base prices (in lakhs)
        city_data = {
            'Mumbai': {'coords': (19.0760, 72.8777), 'base_price': 150},
            'Delhi': {'coords': (28.7041, 77.1025), 'base_price': 120},
            'Bangalore': {'coords': (12.9716, 77.5946), 'base_price': 80},
            'Hyderabad': {'coords': (17.3850, 78.4867), 'base_price': 60},
            'Chennai': {'coords': (13.0827, 80.2707), 'base_price': 70},
            'Pune': {'coords': (18.5204, 73.8567), 'base_price': 75},
            'Kolkata': {'coords': (22.5726, 88.3639), 'base_price': 50},
            'Ahmedabad': {'coords': (23.0225, 72.5714), 'base_price': 45}
        }
        
        city_info = city_data.get(location, city_data['Mumbai'])
        base_lat, base_lon = city_info['coords']
        base_price_lakhs = city_info['base_price']
        area_multiplier = self.get_area_multiplier(location, area)
        
        data = []
        for i in range(num_properties):
            # Generate property features
            bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.15, 0.05])
            bathrooms = np.random.choice([1, 2, 3, 4], p=[0.2, 0.5, 0.25, 0.05])
            sqft = np.random.normal(800 + bedrooms * 200, 200)
            age = np.random.randint(0, 30)
            
            # Location with some clustering around the specific area
            lat = base_lat + np.random.normal(0, 0.02)
            lon = base_lon + np.random.normal(0, 0.02)
            
            # Price calculation in lakhs with area-specific multiplier
            base_price = base_price_lakhs + bedrooms * 15 + bathrooms * 8 + (sqft/100) * 5
            location_factor = np.random.uniform(0.8, 1.2) * area_multiplier
            age_factor = max(0.6, 1 - age * 0.015)
            
            price_lakhs = base_price * location_factor * age_factor
            
            # Future price prediction
            market_growth = np.random.normal(0.08, 0.03)  # 8% average growth for Indian real estate
            future_price = price_lakhs * (1 + market_growth)
            
            data.append({
                'lat': lat,
                'lon': lon,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft': sqft,
                'age': age,
                'area': area,
                'current_price': price_lakhs,
                'predicted_price': future_price,
                'price_change': (future_price - price_lakhs) / price_lakhs * 100
            })
        
        return pd.DataFrame(data)
    
    def predict_property_value(self, bedrooms, bathrooms, sqft, age, location, area):
        """Predict individual property value in lakhs for specific area"""
        # Generate training data
        training_data = self.generate_sample_data(location, area, 1000)
        
        # Prepare features
        X = training_data[['bedrooms', 'bathrooms', 'sqft', 'age']]
        y = training_data['current_price']
        
        # Train model
        self.model.fit(X, y)
        
        # Make prediction
        features = np.array([[bedrooms, bathrooms, sqft, age]])
        prediction = self.model.predict(features)[0]
        
        # Apply area-specific adjustments
        area_multiplier = self.get_area_multiplier(location, area)
        prediction *= area_multiplier
        
        # Calculate future value
        market_growth = np.random.normal(0.08, 0.02)
        future_value = prediction * (1 + market_growth)
        
        return {
            'current_value': prediction,
            'future_value': future_value,
            'expected_growth': market_growth * 100,
            'area_premium': (area_multiplier - 1) * 100
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">💹 AI MARKET PREDICTOR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered predictions for Indian markets</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎯 **SETTINGS**")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 STOCKS", "🏢 REAL ESTATE"])
    
    with tab1:
        st.markdown("### **Stock Price Forecasting**")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("#### **Parameters**")
            
            # Popular Indian stocks
            popular_stocks = {
                "Reliance Industries": "RELIANCE",
                "Tata Consultancy Services": "TCS",
                "HDFC Bank": "HDFCBANK",
                "Infosys": "INFY",
                "ICICI Bank": "ICICIBANK",
                "State Bank of India": "SBIN",
                "Bharti Airtel": "BHARTIARTL",
                "ITC Limited": "ITC",
                "Hindustan Unilever": "HINDUNILVR",
                "Maruti Suzuki": "MARUTI"
            }
            
            selected_stock = st.selectbox("Select Stock", list(popular_stocks.keys()))
            symbol = popular_stocks[selected_stock]
            
            days_ahead = st.slider("Prediction Days", 1, 90, 30)
            period = st.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
            
            predict_button = st.button("🚀 **PREDICT**", type="primary")
        
        with col1:
            if predict_button and symbol:
                stock_predictor = StockPredictor()
                
                # Fetch data
                data, info = stock_predictor.fetch_stock_data(symbol, period)
                
                if data is not None and not data.empty:
                    # Display stock info
                    st.markdown(f"### **{selected_stock}** ({symbol}.NS)")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    current_price = data["Close"][-1]
                    change = ((data["Close"][-1] - data["Close"][-2]) / data["Close"][-2]) * 100
                    
                    with col_a:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-value">₹{current_price:.2f}</div>
                            <div class="metric-label">Current Price</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_b:
                        change_color = "#00ff88" if change >= 0 else "#ff4757"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-value" style="color: {change_color}">{change:+.2f}%</div>
                            <div class="metric-label">Daily Change</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_c:
                        volume = data["Volume"][-1] if "Volume" in data.columns else 0
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-value">{volume:,.0f}</div>
                            <div class="metric-label">Volume</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_d:
                        high_52w = data["High"].max()
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-value">₹{high_52w:.2f}</div>
                            <div class="metric-label">52W High</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Generate predictions
                    predictions = stock_predictor.predict_stock_price(data, days_ahead)
                    
                    # Create prediction chart with dark theme
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Price Analysis & Predictions', 'Confidence Intervals'),
                        vertical_spacing=0.12,
                        row_heights=[0.75, 0.25]
                    )
                    
                    # Historical data
                    fig.add_trace(
                        go.Scatter(
                            x=data.index[-90:],
                            y=data['Close'][-90:],
                            name='Historical',
                            line=dict(color='#666', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Predictions
                    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_ahead)
                    pred_values = [p['prediction'] for p in predictions]
                    upper_bounds = [p['upper'] for p in predictions]
                    lower_bounds = [p['lower'] for p in predictions]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=pred_values,
                            name='AI Prediction',
                            line=dict(color='#00ff88', width=3)
                        ),
                        row=1, col=1
                    )
                    
                    # Confidence intervals
                    fig.add_trace(
                        go.Scatter(
                            x=list(future_dates) + list(future_dates[::-1]),
                            y=upper_bounds + lower_bounds[::-1],
                            fill='toself',
                            fillcolor='rgba(0,255,136,0.1)',
                            line=dict(color='rgba(0,255,136,0)'),
                            name='Confidence Band',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Uncertainty chart
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=[(p['upper'] - p['lower']) / p['prediction'] * 100 for p in predictions],
                            name='Uncertainty',
                            line=dict(color='#ff6b6b'),
                            fill='tozeroy',
                            fillcolor='rgba(255,107,107,0.2)'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        height=700,
                        plot_bgcolor='#0a0a0a',
                        paper_bgcolor='#0a0a0a',
                        font_color='#ffffff',
                        title=f"{selected_stock} - {days_ahead} Day Forecast",
                        title_font_size=20,
                        title_font_color='#00ff88'
                    )
                    
                    fig.update_xaxes(gridcolor='#333', showgrid=True)
                    fig.update_yaxes(gridcolor='#333', showgrid=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction summary
                    predicted_price = predictions[-1]['prediction']
                    expected_return = ((predicted_price - current_price) / current_price) * 100
                    
                    st.markdown(f'''
                    <div class="prediction-box">
                        <div class="prediction-title">AI PREDICTION SUMMARY</div>
                        <div class="prediction-item">Current Price: <span class="prediction-value">₹{current_price:.2f}</span></div>
                        <div class="prediction-item">Target Price ({days_ahead}d): <span class="prediction-value">₹{predicted_price:.2f}</span></div>
                        <div class="prediction-item">Expected Return: <span class="prediction-value">{expected_return:+.2f}%</span></div>
                        <div class="prediction-item">Confidence Range: <span class="prediction-value">₹{predictions[-1]['lower']:.2f} - ₹{predictions[-1]['upper']:.2f}</span></div>
                    </div>
                    ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### **Real Estate Analysis**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### **Property Details**")
            
            indian_cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
            location = st.selectbox("City", indian_cities)
            
            # Get areas for selected city
            re_predictor = RealEstatePredictor()
            city_areas = re_predictor.get_city_areas()
            areas = city_areas.get(location, ["Select City First"])
            area = st.selectbox("Area", areas)
            
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
            sqft = st.number_input("Area (sq ft)", min_value=300, max_value=3000, value=1000)
            age = st.slider("Property Age (years)", 0, 30, 5)
            
            analyze_button = st.button("🏢 **ANALYZE**", type="primary")
        
        with col2:
            if analyze_button and area != "Select City First":
                # Get property prediction
                prediction = re_predictor.predict_property_value(bedrooms, bathrooms, sqft, age, location, area)
                
                # Display prediction results
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">₹{prediction["current_value"]:.1f}L</div>
                        <div class="metric-label">Current Value</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">₹{prediction["future_value"]:.1f}L</div>
                        <div class="metric-label">1Y Projection</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_c:
                    growth_color = "#00ff88" if prediction["expected_growth"] > 0 else "#ff4757"
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value" style="color: {growth_color}">{prediction["expected_growth"]:+.1f}%</div>
                        <div class="metric-label">Growth Rate</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Area premium indicator
                if prediction["area_premium"] != 0:
                    premium_color = "#00ff88" if prediction["area_premium"] > 0 else "#ffa502"
                    premium_text = "Premium" if prediction["area_premium"] > 0 else "Discount"
                    st.markdown(f'''
                    <div style="text-align: center; margin: 1rem 0; padding: 0.5rem; background: #1a1a1a; border-radius: 8px; border: 1px solid {premium_color};">
                        <span style="color: {premium_color}; font-weight: 600;">
                            {area} Area {premium_text}: {abs(prediction["area_premium"]):.1f}%
                        </span>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Generate market data for visualization
                market_data = re_predictor.generate_sample_data(location, area, 150)
                
                # Price per sqft analysis
                fig_analysis = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(f'Price Distribution - {area}', 'Area vs Price Analysis'),
                    specs=[[{"type": "histogram"}, {"type": "scatter"}]]
                )
                
                fig_analysis.add_trace(
                    go.Histogram(
                        x=market_data['current_price'],
                        name='Price Distribution',
                        marker_color='#00ff88',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                fig_analysis.add_trace(
                    go.Scatter(
                        x=market_data['sqft'],
                        y=market_data['current_price'],
                        mode='markers',
                        marker=dict(
                            color=market_data['price_change'],
                            colorscale='RdYlGn',
                            size=8,
                            colorbar=dict(title="Growth %")
                        ),
                        name='Properties',
                        text=[f"Age: {age}yr" for age in market_data['age']],
                        hovertemplate="<b>%{text}</b><br>Area: %{x} sqft<br>Price: ₹%{y:.1f}L<extra></extra>"
                    ),
                    row=1, col=2
                )
                
                fig_analysis.update_layout(
                    height=400,
                    plot_bgcolor='#0a0a0a',
                    paper_bgcolor='#0a0a0a',
                    font_color='#ffffff',
                    title=f"Market Analysis - {area}, {location}",
                    title_font_color='#00ff88'
                )
                
                fig_analysis.update_xaxes(gridcolor='#333')
                fig_analysis.update_yaxes(gridcolor='#333')
                
                st.plotly_chart(fig_analysis, use_container_width=True)
                
                # Investment recommendation
                if prediction["expected_growth"] > 8:
                    recommendation = "🟢 **STRONG BUY** - Excellent growth potential"
                    rec_color = "#00ff88"
                elif prediction["expected_growth"] > 4:
                    recommendation = "🟡 **HOLD** - Moderate growth expected"
                    rec_color = "#ffa502"
                else:
                    recommendation = "🔴 **CAUTION** - Limited growth potential"
                    rec_color = "#ff4757"
                
                st.markdown(f'''
                <div class="prediction-box">
                    <div class="prediction-title">INVESTMENT ANALYSIS</div>
                    <div class="prediction-item" style="color: {rec_color}; font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem;">{recommendation}</div>
                    <div class="prediction-item">Location: <span class="prediction-value">{area}, {location}</span></div>
                    <div class="prediction-item">Property: <span class="prediction-value">{bedrooms}BHK, {bathrooms} Bath</span></div>
                    <div class="prediction-item">Rate: <span class="prediction-value">₹{prediction["current_value"]*100000/sqft:,.0f} per sq ft</span></div>
                    <div class="prediction-item">Annual ROI: <span class="prediction-value">{prediction["expected_growth"]:.1f}%</span></div>
                </div>
                ''', unsafe_allow_html=True)
            
            elif analyze_button:
                st.warning("Please select a valid area to analyze the property.")
        
        # Market overview
        st.markdown("### **Market Overview**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # City price comparison
            cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"]
            avg_prices = [150, 120, 80, 60, 70]
            
            fig_cities = go.Figure(data=[
                go.Bar(
                    x=cities,
                    y=avg_prices,
                    marker_color='#00ff88',
                    marker_line_color='#00cc6a',
                    marker_line_width=2
                )
            ])
            
            fig_cities.update_layout(
                title="Avg Price (₹L per property)",
                height=300,
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#0a0a0a',
                font_color='#ffffff',
                title_font_color='#00ff88'
            )
            
            fig_cities.update_xaxes(gridcolor='#333')
            fig_cities.update_yaxes(gridcolor='#333')
            
            st.plotly_chart(fig_cities, use_container_width=True)
        
        with col2:
            # Interest rate trends
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            rates = [6.5, 6.7, 6.9, 7.1, 7.0, 6.8]
            
            fig_rates = go.Figure()
            fig_rates.add_trace(go.Scatter(
                x=months,
                y=rates,
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8)
            ))
            
            fig_rates.update_layout(
                title="Home Loan Rates (%)",
                height=300,
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#0a0a0a',
                font_color='#ffffff',
                title_font_color='#00ff88'
            )
            
            fig_rates.update_xaxes(gridcolor='#333')
            fig_rates.update_yaxes(gridcolor='#333')
            
            st.plotly_chart(fig_rates, use_container_width=True)
        
        with col3:
            # Market sentiment
            sentiment_data = ['Bullish', 'Neutral', 'Bearish']
            sentiment_values = [45, 35, 20]
            colors = ['#00ff88', '#ffa502', '#ff4757']
            
            fig_sentiment = go.Figure(data=[
                go.Pie(
                    labels=sentiment_data,
                    values=sentiment_values,
                    marker_colors=colors,
                    hole=0.4
                )
            ])
            
            fig_sentiment.update_layout(
                title="Market Sentiment",
                height=300,
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#0a0a0a',
                font_color='#ffffff',
                title_font_color='#00ff88'
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-title">🤖 POWERED BY ADVANCED AI</div>
        <div class="disclaimer">
            This application uses LSTM neural networks for stock predictions and Random Forest algorithms for real estate analysis.<br>
            <strong>Disclaimer:</strong> All predictions are for educational purposes only. Market investments carry inherent risks.<br>
            Always consult certified financial advisors before making investment decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()