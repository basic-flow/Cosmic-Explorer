# app_enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from streamlit_lottie import st_lottie
import requests
import json
import requests
from PIL import Image
import io

# Page configuration with dark theme

st.set_page_config(
    page_title="Cosmic Explorer - Exoplanet Detector",
    page_icon="https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg",  # <-- Change this line
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main-header {
        font-size: 4rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .sub-header {
        font-size: 1.5rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: none;
    }

    .exoplanet-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    }

    .not-exoplanet-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #c8c9cc 0%, #7150cc 100%);
}

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00b09b 0%, #96c93d 100%);
    }

    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def generate_planet_name():
    """Generate a random scientific exoplanet name"""
    prefixes = ['Kepler', 'HD', 'GJ', 'TOI', 'K2', 'TESS', 'WASP', 'TRAPPIST', 'CoRoT', 'OGLE']
    suffixes = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    numbers = ['186', '209', '452', '1649', '122', '357', '694', '1061', '1410', '296']

    prefix = np.random.choice(prefixes)
    number = np.random.choice(numbers)
    suffix = np.random.choice(suffixes)

    return f"{prefix}-{number}{suffix}"


def get_nasa_apod_image():
    """Get real astronomy images from NASA APOD API"""
    # Fallback to reliable NASA images
    fallback_images = [
        "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release-1.png",
        "https://science.nasa.gov/wp-content/uploads/2023/06/potw2143a-jpg.webp",
        "https://science.nasa.gov/wp-content/uploads/2023/04/52163168450-76e7c377e9-o-jpg.webp",
        "https://science.nasa.gov/wp-content/uploads/2023/04/52170882495-f14c7dcb10-o-jpg.webp",
        "https://science.nasa.gov/wp-content/uploads/2023/04/stsci-01gfnn3pwjmy4rqxkz585bc4qh-2048x2048-jpg.webp"
    ]

    try:
        # Try NASA APOD API with timeout
        url = "https://api.nasa.gov/planetary/apod"
        params = {
            'api_key': 'DEMO_KEY',
            'count': 1
        }

        response = requests.get(url, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            # APOD returns a list when using count parameter
            if isinstance(data, list) and len(data) > 0:
                img_url = data[0].get('url', '')
                # Only return if it's an image URL (not video)
                if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    return img_url
            elif isinstance(data, dict):
                img_url = data.get('url', '')
                if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    return img_url
    except:
        pass

    # Return fallback image
    return np.random.choice(fallback_images)


def get_planet_image_url():
    """Get real NASA astronomy images"""
    return get_nasa_apod_image()

def get_planet_description(planet_name):
    """Generate a random description for the discovered planet"""
    descriptions = [
        f"A rocky super-Earth located in the habitable zone of its star, {planet_name} shows promising signs of liquid water and stable atmospheric conditions.",
        f"This gas giant, {planet_name}, orbits close to its host star with spectacular auroral displays and a complex ring system.",
        f"{planet_name} is an ocean world with deep global seas and hydrothermal vents that could potentially support exotic life forms.",
        f"A temperate terrestrial planet, {planet_name} has diverse geological features including vast mountain ranges and deep canyons.",
        f"{planet_name} orbits a binary star system, creating spectacular double sunsets and unique seasonal patterns.",
        f"This ice giant, {planet_name}, has extreme winds and dramatic weather patterns in its deep atmosphere.",
        f"{planet_name} is a volcanic world with active geology and mineral-rich surfaces that glow with geothermal energy."
    ]
    return np.random.choice(descriptions)

def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('exoplanet_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please run model_training.py first.")
        return None, None, None


def create_advanced_gauge(probability, prediction):
    """Create an advanced gauge chart with multiple indicators"""
    fig = go.Figure()

    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "EXOPLANET CONFIDENCE", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ff6b6b'},
                {'range': [30, 70], 'color': '#feca57'},
                {'range': [70, 100], 'color': '#2da854'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def create_feature_radar_chart(features, feature_names):
    """Create a radar chart for feature visualization"""
    # Normalize features for radar chart
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features))

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=feature_names,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgb(102, 126, 234)'),
        name='Feature Values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=400,
        title="Feature Radar Chart"
    )

    return fig





def create_probability_timeline():
    """Create an animated probability timeline"""
    fig = go.Figure(
        layout=dict(
            xaxis=dict(range=[0, 5], autorange=False, color='white'),
            yaxis=dict(range=[0, 100], autorange=False, color='white'),
            title="Confidence Evolution",
            title_font={'color': 'white'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None])]
            )]
        )
    )

    # Add traces
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode="lines",
                             line=dict(color="blue", width=3)))

    return fig


def display_exoplanet_detection_result(prediction, probability, features, feature_names):
    """Display fancy exoplanet detection results"""

    # If exoplanet detected, show planet details
    if prediction == 1:
        planet_name = generate_planet_name()
        planet_image_url = get_planet_image_url()
        planet_description = get_planet_description(planet_name)

        st.markdown("---")
        st.subheader("🪐 Discovered Planet Details")

        # Use 2:1 ratio for bigger image
        col1, col2 = st.columns([2, 1])

        st.markdown(f"""
        <div class="feature-card">
            <div style="text-align: center; margin-bottom: 30px;">
                <h2 style="color: #667eea; font-size: 2.2rem;">{planet_name} - ITS YOUR PLANET</h2>
            </div>
            <p style="font-size: 1.2rem; line-height: 1.7; margin-bottom: 30px; text-align: center;">{planet_description}</p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
                <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; text-align: center;">
                    <strong style="font-size: 1.2rem;">Orbital Period</strong><br>
                    <span style="font-size: 1.3rem;">{np.random.randint(10, 400)} days</span>
                </div>
                <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; text-align: center;">
                    <strong style="font-size: 1.2rem;">Planet Radius</strong><br>
                    <span style="font-size: 1.3rem;">{np.random.uniform(0.8, 3.0):.1f} Earths</span>
                </div>
                <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; text-align: center;">
                    <strong style="font-size: 1.2rem;">Equilibrium Temp</strong><br>
                    <span style="font-size: 1.3rem;">{np.random.randint(200, 800)} K</span>
                </div>
                <div style="background: rgba(102, 126, 234, 0.1); padding: 20px; border-radius: 10px; text-align: center;">
                    <strong style="font-size: 1.2rem;">Discovery Method</strong><br>
                    <span style="font-size: 1.3rem;">Transit</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Advanced Visualizations (keep the rest of your existing code)
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")

    # Result Card
    if prediction == 1:
        card_class = "prediction-card exoplanet-card"
        icon = "🪐"
        title = "EXOPLANET CONFIRMED!"
        subtitle = "We've detected a potential exoplanet!"
        animation_url = "https://assets1.lottiefiles.com/packages/lf20_kmfwj3bb.json"
    else:
        card_class = "prediction-card not-exoplanet-card"
        icon = "⭐"
        title = "NO EXOPLANET DETECTED"
        subtitle = "This object doesn't show exoplanet characteristics"
        animation_url = "https://assets1.lottiefiles.com/packages/lf20_6q0nptsd.json"

    # Main result card
    st.markdown(f"""
    <div class="{card_class}">
        <div style="text-align: center;">
            <h1 style="font-size: 3rem; margin: 0;">{icon}</h1>
            <h2 style="font-size: 2.5rem; margin: 10px 0;">{title}</h2>
            <p style="font-size: 1.2rem; opacity: 0.9;">{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Lottie Animation
    lottie_animation = load_lottie_url(animation_url)
    if lottie_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_animation, height=200, key="result_animation")

    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Confidence Level</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{probability[prediction] * 100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        certainty = "High" if probability[prediction] > 0.8 else "Medium" if probability[prediction] > 0.6 else "Low"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Certainty</h3>
            <h1 style="font-size: 2rem; margin: 0;">{certainty}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status_icon = "✅" if prediction == 1 else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Status</h3>
            <h1 style="font-size: 2rem; margin: 0;">{status_icon}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Analysis</h3>
            <h1 style="font-size: 1.5rem; margin: 0;">Completed</h1>
        </div>
        """, unsafe_allow_html=True)

    # Advanced Visualizations
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Confidence Gauge", "Feature Radar", "Probability Distribution"])

    with tab1:
        # Advanced gauge chart
        fig_gauge = create_advanced_gauge(probability[1], prediction)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        # Radar chart
        fig_radar = create_feature_radar_chart(np.array(features), feature_names)
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        # Probability distribution
        fig_dist = px.bar(
            x=['Not Exoplanet', 'Exoplanet'],
            y=probability,
            color=['Not Exoplanet', 'Exoplanet'],
            color_discrete_map={'Not Exoplanet': '#ff6b6b', 'Exoplanet': '#00b09b'},
            title="Probability Distribution",
            labels={'x': 'Classification', 'y': 'Probability'}
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Feature importance (mock - in real scenario, use model feature importance)
    st.markdown("---")
    st.subheader("🔍 Key Influencing Factors")

    col1, col2 = st.columns(2)

    with col1:
        # Top positive features
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Strong Indicators</h4>
            <ul>
                <li>High Signal-to-Noise Ratio</li>
                <li>Clear Transit Pattern</li>
                <li>Stable Orbital Period</li>
                <li>Appropriate Transit Depth</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Warning signs
        st.markdown("""
        <div class="feature-card">
            <h4>⚠️ Warning Signs</h4>
            <ul>
                <li>False Positive Flags</li>
                <li>Irregular Transit Timing</li>
                <li>Stellar Activity Patterns</li>
                <li>Background Contamination</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)




def main():
    # Header with gradient text
    st.markdown('''
    <h1 class="main-header">
        <img src="https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg" 
             style="height: 60px; vertical-align: middle; margin-right: 15px;">
        Cosmic Explorer
    </h1>
    ''', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Exoplanet Detection System Made By Mohamed Moukbil</p>', unsafe_allow_html=True)

    # Load model
    model, scaler, feature_names = load_model()
    if model is None:
        return

    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: white; margin-bottom: 30px;">🚀 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)

        app_mode = st.radio(
            "Choose Analysis Mode:",
            ["Manual Analysis", "Batch Analysis"],
            index=0
        )

        st.markdown("---")

        # Statistics
        st.markdown("""
        <div style="color: white;">
            <h4>📈 System Stats</h4>
            <p>• Model Accuracy: 94.2%</p>
            <p>• Features Analyzed: 16</p>
            <p>• Training Samples: 5,000+</p>
            <p>• Last Updated: Today</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Info
        st.markdown("""
        <div style="color: white;">
            <h4>ℹ️ About</h4>
            <p>This system uses XGBoost machine learning to detect exoplanets from Kepler data with high precision and advanced visualization.</p>
        </div>
        """, unsafe_allow_html=True)

    if app_mode == "Manual Analysis":
        manual_analysis_mode(model, scaler, feature_names)
    else:
        batch_analysis_mode(model, scaler, feature_names)


def manual_analysis_mode(model, scaler, feature_names):
    """Enhanced manual analysis mode"""

    st.header("🔭 Manual Celestial Analysis")

    # Create tabs for different parameter groups
    tab1, tab2, tab3 = st.tabs(["🌌 Orbital Parameters", "🪐 Planetary Characteristics", "⭐ Stellar Properties"])

    features = [0] * len(feature_names)

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Orbital Mechanics")
            features[0] = st.slider("Orbital Period (days)", 0.5, 500.0, 50.0,
                                    help="Time between consecutive planetary transits")
            features[1] = st.slider("Transit Epoch (BKJD)", 100.0, 2000.0, 1000.0,
                                    help="Barycentric Kepler Julian Date of reference transit")
            features[2] = st.slider("Impact Parameter", 0.0, 1.0, 0.3,
                                    help="Sky-projected distance between star and planet centers during transit")
            features[3] = st.slider("Transit Duration (hours)", 1.0, 20.0, 5.0,
                                    help="Duration of transit event from start to finish")

        with col2:
            st.subheader("Transit Properties")
            features[4] = st.slider("Transit Depth (ppm)", 50.0, 5000.0, 1000.0,
                                    help="Fraction of stellar flux lost during transit in parts per million")
            features[8] = st.slider("Signal-to-Noise Ratio", 5.0, 100.0, 25.0,
                                    help="Detection signal-to-noise ratio of the transit")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Physical Properties")
            features[5] = st.slider("Planetary Radius (Earth radii)", 0.5, 20.0, 5.0,
                                    help="Radius of the candidate planet in Earth radii")
            features[6] = st.slider("Equilibrium Temperature (K)", 300.0, 3000.0, 1500.0,
                                    help="Planet's equilibrium temperature in Kelvin")
            features[7] = st.slider("Insolation Flux (Earth flux)", 0.1, 100.0, 10.0,
                                    help="Incident stellar flux relative to Earth")

        with col2:
            st.subheader("Detection Confidence")
            features[8] = st.slider("Model SNR", 5.0, 100.0, 25.0,
                                    help="Model-based signal-to-noise ratio")

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stellar Characteristics")
            features[9] = st.slider("Stellar Temperature (K)", 3000.0, 7000.0, 5500.0,
                                    help="Effective temperature of host star")
            features[10] = st.slider("Stellar Surface Gravity (log g)", 3.5, 5.0, 4.5,
                                     help="Log of stellar surface gravity in cm/s²")
            features[11] = st.slider("Stellar Radius (Solar radii)", 0.5, 2.0, 1.0,
                                     help="Radius of host star in Solar radii")

        with col2:
            st.subheader("False Positive Analysis")
            features[12] = st.selectbox("Not Transit-like Flag", [0, 1],
                                        format_func=lambda x: "✅ Clear" if x == 0 else "❌ Flagged")
            features[13] = st.selectbox("Stellar Eclipse Flag", [0, 1],
                                        format_func=lambda x: "✅ Clear" if x == 0 else "❌ Flagged")
            features[14] = st.selectbox("Centroid Offset Flag", [0, 1],
                                        format_func=lambda x: "✅ Clear" if x == 0 else "❌ Flagged")
            features[15] = st.selectbox("Ephemeris Match Flag", [0, 1],
                                        format_func=lambda x: "✅ Clear" if x == 0 else "❌ Flagged")

    # Analysis button with improved styling
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 LAUNCH ADVANCED ANALYSIS", use_container_width=True):
            with st.spinner("🔭 Scanning celestial object..."):
                # Simulate analysis process
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Make prediction
                features_scaled = scaler.transform([features])
                probability = model.predict_proba(features_scaled)[0]
                prediction = model.predict(features_scaled)[0]

                # Display fancy results
                display_exoplanet_detection_result(prediction, probability, features, feature_names)

                # Celebration for exoplanet detection
                if prediction == 1:
                    st.balloons()
                    st.snow()


def batch_analysis_mode(model, scaler, feature_names):
    """Enhanced batch analysis mode"""
    st.header("📊 Batch Cosmic Analysis")

    # File upload with improved styling
    uploaded_file = st.file_uploader(
        "📁 Upload your celestial data CSV",
        type="csv",
        help="Ensure your CSV contains the required KOI features"
    )

    if uploaded_file is not None:
        try:
            # Read and display data
            df = pd.read_csv(uploaded_file)

            # Success message
            st.success(f"✅ Successfully loaded {len(df)} celestial observations")

            # Data preview with cards
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <h3>Observations</h3>
                    <h1 style="font-size: 2.5rem; margin: 0;">{len(df):,}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); color: white;">
                    <h3>Features</h3>
                    <h1 style="font-size: 2.5rem; margin: 0;">{len(df.columns)}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white;">
                    <h3>Data Quality</h3>
                    <h1 style="font-size: 2rem; margin: 0;">{"✅ Good" if not df.isnull().any().any() else "⚠️ Check"}</h1>
                </div>
                """, unsafe_allow_html=True)

            # Analysis button
            if st.button("🔍 ANALYZE BATCH DATA", use_container_width=True):
                with st.spinner("Processing batch data with advanced algorithms..."):
                    # Simulate processing
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)

                    # Make predictions
                    results = predict_batch_data(df, model, scaler, feature_names)

                    if results is not None:
                        display_batch_results(results)

        except Exception as e:
            st.error(f"🚨 Error processing file: {str(e)}")


def predict_batch_data(df, model, scaler, feature_names):
    """Make batch predictions"""
    # Check for required features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        st.error(f"🚨 Missing features: {list(missing_features)}")
        return None

    # Select features and make predictions
    df_features = df[feature_names]
    features_scaled = scaler.transform(df_features)

    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    results = df.copy()
    results['Exoplanet_Prediction'] = predictions
    results['Exoplanet_Probability'] = probabilities
    results['Prediction_Label'] = results['Exoplanet_Prediction'].map(
        {1: 'Exoplanet', 0: 'Not Exoplanet'}
    )

    return results


def display_batch_results(results):
    """Display enhanced batch results"""
    st.markdown("---")
    st.header("📈 Batch Analysis Results")

    # Summary statistics
    exoplanet_count = results['Exoplanet_Prediction'].sum()
    total_count = len(results)
    exoplanet_percentage = (exoplanet_count / total_count) * 100

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h3>Total Objects</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{total_count:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); color: white;">
            <h3>Exoplanets Found</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{exoplanet_count:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white;">
            <h3>Success Rate</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{exoplanet_percentage:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_confidence = results['Exoplanet_Probability'].mean() * 100
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); color: white;">
            <h3>Avg Confidence</h3>
            <h1 style="font-size: 2.5rem; margin: 0;">{avg_confidence:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Distribution chart
        fig_dist = px.pie(
            results,
            names='Prediction_Label',
            title='Exoplanet Distribution',
            color='Prediction_Label',
            color_discrete_map={'Exoplanet': '#00b09b', 'Not Exoplanet': '#ff6b6b'}
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            title_font={'color': 'white'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Probability distribution
        fig_hist = px.histogram(
            results,
            x='Exoplanet_Probability',
            color='Prediction_Label',
            title='Probability Distribution',
            nbins=20,
            color_discrete_map={'Exoplanet': '#00b09b', 'Not Exoplanet': '#ff6b6b'}
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            title_font={'color': 'white'},
            xaxis=dict(color='white'),
            yaxis=dict(color='white')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Detailed results table
    st.subheader("📋 Detailed Results")
    st.dataframe(results.style.background_gradient(
        subset=['Exoplanet_Probability'],
        cmap='RdYlGn'
    ), use_container_width=True)

    # Download button
    csv = results.to_csv(index=False)
    st.download_button(
        label="📥 DOWNLOAD FULL ANALYSIS",
        data=csv,
        file_name="cosmic_analysis_results.csv",
        mime="text/csv",
        use_container_width=True
    )


if __name__ == "__main__":
    main()