import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ---- FIXED: Generate consistent data ----
np.random.seed(42)  # Set seed for reproducibility

base_lat = 19.0760
base_lon = 72.8777
grid_data = []

for i in range(10):
    for j in range(10):
        lat = base_lat + i * 0.001
        lon = base_lon + j * 0.001
        hour = np.random.randint(0, 24)
        dist_to_main_road = abs(lat - (base_lat + 0.005))

        noise_db = np.random.randint(40, 90)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            noise_db += 20
        elif 22 <= hour or hour <= 5:
            noise_db -= 10
        noise_db -= dist_to_main_road * 30
        noise_db = min(max(noise_db, 30), 100)

        grid_data.append([lat, lon, hour, dist_to_main_road, noise_db])

df = pd.DataFrame(grid_data, columns=['latitude', 'longitude', 'hour', 'dist_to_main_road', 'noise_db'])

# ---- Train model ----
X = df[['hour', 'dist_to_main_road', 'latitude', 'longitude']]
y = df['noise_db']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---- Streamlit UI ----
st.set_page_config(page_title="Urban Noise Predictor", layout="centered")
st.title("🌆 Urban Noise Level Predictor")
st.markdown("Enter a location and time to predict noise level, and see a simulated city noise heatmap.")

# User input
latitude = st.number_input("📍 Enter Latitude", value=19.080, format="%.6f")
longitude = st.number_input("📍 Enter Longitude", value=72.880, format="%.6f")
hour = st.slider("⏰ Select Hour of Day", 0, 23, 12)

# Calculate distance to main road
main_road_lat = base_lat + 0.005
dist_to_main_road = abs(latitude - main_road_lat)

# Predict noise level
input_data = pd.DataFrame([[hour, dist_to_main_road, latitude, longitude]],
                          columns=['hour', 'dist_to_main_road', 'latitude', 'longitude'])
predicted_noise = model.predict(input_data)[0]

# Show prediction
st.subheader("🔊 Predicted Noise Level")
st.write(f"**{predicted_noise:.2f} dB**")

# Show category message
st.markdown("### 📊 Noise Category")
if predicted_noise >= 80:
    st.error("🔴 **Very Noisy Area** — High chance of traffic, construction, or crowds.")
elif predicted_noise >= 60:
    st.warning("🟠 **Moderate Noise** — Normal urban sound levels.")
else:
    st.success("🟢 **Quiet Area** — Peaceful environment.")

# ---- Heatmap Section ----
st.subheader("🌍 Simulated Noise Heatmap (Static)")

# Create Folium map centered on the grid
m = folium.Map(location=[base_lat + 0.005, base_lon + 0.005], zoom_start=15)

# Prepare heatmap data
heat_data = [[row['latitude'], row['longitude'], row['noise_db']] for _, row in df.iterrows()]

# Add heatmap
HeatMap(heat_data, radius=15).add_to(m)

# Show map in Streamlit
st_folium(m, width=700, height=500)
