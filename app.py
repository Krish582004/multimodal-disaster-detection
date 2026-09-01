import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from src.pipeline import search_satellite_data
from src.preprocess import generate_multimodal_tensor
from src.model import run_vision_inference
from src.nlp_module import get_text_disaster_score
from src.fusion import generate_final_payload

# --- Page Config ---
st.set_page_config(page_title="AI Disaster Monitor", layout="wide")
st.title("🌍 Multimodal Satellite Disaster Detection")

# --- Initialize Session State ---
# This tells Streamlit to remember the AI results even after the page refreshes
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# --- Sidebar Controls ---
st.sidebar.header("Analysis Parameters")

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
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

folium.Rectangle(
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    color='#3388ff',
    fill=False,
    dash_array='5, 5',
    tooltip='Scanning Region'
).add_to(m)

# 1. Run the analysis ONLY when the button is clicked
if run_analysis:
    with st.spinner("Querying satellites and running AI models..."):
        search_satellite_data(bbox, days_back=30)
        tensor = generate_multimodal_tensor(bbox)
        vis_score = run_vision_inference(tensor)
        nlp_score = get_text_disaster_score(bbox, target_hazard=disaster_type)
        
        # Save the final payload into memory instead of just a local variable
        st.session_state.analysis_result = generate_final_payload(bbox, disaster_type, vis_score, nlp_score)

# 2. Render the UI continuously as long as data exists in memory
if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    
    if result["metrics"]["alert_triggered"]:
        st.error(f"🚨 **ALERT:** High probability of {result['disaster_type']} detected!")
    else:
        st.success("✅ No severe hazards detected in this region.")
        
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Visual AI Confidence", f"{result['metrics']['visual_confidence']*100:.1f}%")
    m_col2.metric("NLP Text Confidence", f"{result['metrics']['nlp_confidence']*100:.1f}%")
    m_col3.metric("Fused Alert Score", f"{result['metrics']['fused_score']*100:.1f}%")
    
    geojson_data = result["spatial_features"]
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': '#ff0000' if feature['properties']['severity'] == 'HIGH' else '#ffa500',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=['hazard', 'severity'])
    ).add_to(m)

# Render the map
st_folium(m, width=1000, height=500)