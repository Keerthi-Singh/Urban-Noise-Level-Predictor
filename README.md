# Urban-Noise-Level-Predictor

A Streamlit web app that simulates and predicts urban noise levels based on user input like location and time. Ideal for visualizing noise distribution in city environments.

🔗 **Live App**: [urban-noise-level-predictor-keerthisingh.streamlit.app](https://urban-noise-level-predictor-keerthisingh.streamlit.app/)

---

## Features

- Predict noise level (in dB) based on:
  - Latitude
  - Longitude
  - Hour of the day
- Classify noise into zones: Quiet / Moderate / Noisy
- Display an interactive **Folium heatmap** of simulated city noise
- Simple, intuitive interface powered by Streamlit

---

## How It Works

1. **Simulated data** is generated for a 10x10 city grid (lat/lon).
2. **Noise levels** are influenced by:
   - Distance from the main road
   - Time of day (rush hours vs night)
3. A **Random Forest Regressor** model is trained on this simulated data.
4. The app predicts noise level using user input and displays:
   - Noise category
   - A real-time prediction
   - A city-wide noise heatmap

---


