import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from mock_data import get_mock_prediction

# --- Page Config ---
st.set_page_config(page_title="AI Disaster Monitor", layout="wide")
st.title("🌍 Multimodal Satellite Disaster Detection")

# --- Sidebar Controls ---
st.sidebar.header("Analysis Parameters")

# Default bounding box (Kolkata region as an example)
st.sidebar.markdown("**Bounding Box (Min Lon, Min Lat, Max Lon, Max Lat)**")
col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input("Min Lon", value=88.20)
min_lat = col2.number_input("Min Lat", value=22.45)
max_lon = col1.number_input("Max Lon", value=88.50)
max_lat = col2.number_input("Max Lat", value=22.70)

bbox = [min_lon, min_lat, max_lon, max_lat]
disaster_type = st.sidebar.selectbox("Target Hazard", ["Flood", "Wildfire", "Landslide"])

run_analysis = st.sidebar.button("Run AI Analysis", type="primary")

# --- Main Map Area ---
# Center map on the bounding box
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

# Draw the scanning bounding box
folium.Rectangle(
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    color='#3388ff',
    fill=False,
    dash_array='5, 5',
    tooltip='Scanning Region'
).add_to(m)

if run_analysis:
    with st.spinner("Querying satellites and running AI models..."):
        # Fetch the simulated AI response
        result = get_mock_prediction(bbox, disaster_type=disaster_type)
        
        if result["metrics"]["alert_triggered"]:
            st.error(f"🚨 **ALERT:** High probability of {result['disaster_type']} detected!")
        else:
            st.success("✅ No severe hazards detected in this region.")
            
        # Display metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Visual AI Confidence", f"{result['metrics']['visual_confidence']*100:.1f}%")
        m_col2.metric("NLP Text Confidence", f"{result['metrics']['nlp_confidence']*100:.1f}%")
        m_col3.metric("Fused Alert Score", f"{result['metrics']['fused_score']*100:.1f}%")
        
        # Overlay the detected disaster polygon on the map
        geojson_data = result["spatial_features"]
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': '#ff0000' if feature['properties']['severity'] == 'HIGH' else '#ffa500',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(fields=['hazard', 'severity', 'confidence'])
        ).add_to(m)

# Render the map
st_folium(m, width=1000, height=500)