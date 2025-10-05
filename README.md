
# 🌌 Cosmic Explorer - Exoplanet Detection System

**Developed for the Space App Challenge**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%2520Learning-FF6B6B?style=for-the-badge&logo=ai&logoColor=white)

---
## 🚀 Overview

**Cosmic Explorer** is an advanced web application that uses **machine learning** to detect potential exoplanets from astronomical data. Built with **Streamlit** and powered by **XGBoost**, this interactive platform brings the excitement of exoplanet discovery to astronomers, students, and space enthusiasts alike.

---
## ✨ Features

### 🔭 Interactive Analysis Modes

* **Manual Analysis**: Adjust 16 astrophysical parameters using intuitive sliders
* **Batch Analysis**: Upload CSV files for processing multiple observations
* **Real-time Predictions**: Get instant results with confidence scores

### 📊 Advanced Visualizations

* Interactive gauge charts showing detection confidence
* Radar charts for feature comparison
* Probability distribution graphs
* Batch analysis summary dashboards

### 🪐 Immersive Experience

* NASA **APOD** integration for real astronomy images
* Generated planet names and descriptions
* Celebration animations for exoplanet discoveries
* Dark theme with cosmic styling

---
## 🛠️ Installation

### Prerequisites

* **Python 3.8+**
* **pip** package manager

### Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/cosmic-explorer.git](https://github.com/yourusername/cosmic-explorer.git)
    cd cosmic-explorer
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the machine learning model**
    ```bash
    python model_training.py
    ```

4.  **Launch the application**
    ```bash
    streamlit run app.py
    ```

---
## 🎯 Usage

### Manual Analysis

1.  Select "Manual Analysis" from the sidebar
2.  Adjust parameters across three tabs:
    * 🌌 **Orbital Parameters** (period, duration, impact)
    * 🪐 **Planetary Characteristics** (radius, temperature, flux)
    * ⭐ **Stellar Properties** (temperature, gravity, false positive flags)
3.  Click "🚀 LAUNCH ADVANCED ANALYSIS"
4.  View detailed results with visualizations

### Batch Analysis

1.  Select "Batch Analysis" from the sidebar
2.  Upload a **CSV file** with required KOI features
3.  Click "🔍 ANALYZE BATCH DATA"
4.  Download comprehensive results

---
## 🤖 Machine Learning Model

### Model Architecture

* **Algorithm**: XGBoost Classifier
* **Accuracy**: 94.2% on synthetic test data
* **Features**: 16 Kepler Object of Interest parameters
* **Training Samples**: 5,000+ synthetic observations

### Key Features Analyzed

* Orbital period and transit parameters
* Planetary physical characteristics
* Stellar properties and temperatures
* False positive detection flags

---
## 🌟 Example Output

When an exoplanet is detected:

* 🪐 Unique planet name (e.g., "**Kepler-186b**")
* 📊 Confidence score and visualization
* 🖼️ Real NASA astronomy image
* 📝 Generated planet description
* 🎉 Celebration animations

---
## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
| :--- | :--- |
| Model files not found | Run \`python model_training.py\` first |
| Missing dependencies | Install all packages from \`requirements.txt\` |
| NASA API issues | Application includes fallback images |
| CSV upload errors | Ensure file contains all required features |

---
## 🤝 Contributing

We welcome contributions! Please feel free to submit **pull requests** for:

* New visualization features
* Additional machine learning models
* UI/UX improvements
* Documentation enhancements

---
## 📄 License

This project is licensed under the **MIT License** - see the \`LICENSE\` file for details.

---
## 🙏 Acknowledgments

* NASA for Kepler data and APOD API
* Streamlit team for the amazing framework
* XGBoost developers for the robust ML library
* Plotly for interactive visualizations

---
**Made by Mohamed Moukbil**

Explore the cosmos from your browser! 🌠
