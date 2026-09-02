import streamlit as st
import folium
from streamlit_folium import st_folium
from src.pipeline import search_satellite_data
from src.preprocess import generate_multimodal_tensor
from src.model import run_vision_inference
from src.nlp_module import get_text_disaster_score
from src.fusion import generate_final_payload
from src.sos_router import generate_sos_payload
from src.evacuation import generate_safe_shelters, generate_evacuation_corridors
from src.sms_alert import send_emergency_sms

# --- Page Config ---
st.set_page_config(page_title="Disaster Command Center", layout="wide")
st.title("🏛️ Intelligent Disaster C3 & Evacuation System")
st.markdown("*Autonomous Hazard Detection, Impact Assessment, and Evacuation Routing*")

# --- Initialize Session State ---
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "sos_data" not in st.session_state:
    st.session_state.sos_data = None
if "shelters" not in st.session_state:
    st.session_state.shelters = []
if "corridors" not in st.session_state:
    st.session_state.corridors = []

# --- Sidebar Controls ---
st.sidebar.header("Sector Telemetry")
col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input("Min Lon", value=88.20)
min_lat = col2.number_input("Min Lat", value=22.45)
max_lon = col1.number_input("Max Lon", value=88.50)
max_lat = col2.number_input("Max Lat", value=22.70)

bbox = [min_lon, min_lat, max_lon, max_lat]
disaster_type = st.sidebar.selectbox("Target Hazard", ["flood", "wildfire", "landslide", "earthquake"])

run_analysis = st.sidebar.button("Execute Satellite Scan", type="primary")

# --- Main Map Area ---
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

# Draw the scanning bounding box
folium.Rectangle(
    bounds=[[min_lat, min_lon], [max_lat, max_lon]], 
    color='#3388ff', 
    fill=False,
    dash_array='5, 5',
    tooltip='Scanning Region'
).add_to(m)

# 1. Run Analysis Pipeline
if run_analysis:
    with st.spinner("Analyzing satellite telemetry and generating evacuation corridors..."):
        search_satellite_data(bbox, days_back=30)
        tensor = generate_multimodal_tensor(bbox)
        vis_score = run_vision_inference(tensor)
        nlp_score = get_text_disaster_score(bbox, target_hazard=disaster_type)
        
        result = generate_final_payload(bbox, disaster_type, vis_score, nlp_score)
        st.session_state.analysis_result = result
        
        if result["metrics"]["alert_triggered"]:
            severity = result["spatial_features"]["features"][0]["properties"]["severity"]
            # Generate SOS Payload
            st.session_state.sos_data = generate_sos_payload(
                disaster_type=disaster_type, 
                severity=severity, 
                bbox=bbox, 
                timestamp=result["timestamp"]
            )
            # Generate safe zones & evacuation routes
            st.session_state.shelters = generate_safe_shelters(bbox, disaster_type)
            st.session_state.corridors = generate_evacuation_corridors(bbox, st.session_state.shelters)
        else:
            st.session_state.sos_data = None
            st.session_state.shelters = []
            st.session_state.corridors = []

# 2. Render Results
if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    
    if result["metrics"]["alert_triggered"]:
        st.error(f"🚨 **EMERGENCY DETECTED:** {result['disaster_type'].upper()} — EVACUATION PROTOCOLS ACTIVE")
    else:
        st.success("✅ Sector clear. Normal conditions.")
        
    # Metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Visual AI Confidence", f"{result['metrics']['visual_confidence']*100:.1f}%")
    m_col2.metric("NLP Text Confidence", f"{result['metrics']['nlp_confidence']*100:.1f}%")
    m_col3.metric("Fused Alert Score", f"{result['metrics']['fused_score']*100:.1f}%")
    
    # Overlay Hazard Polygon
    folium.GeoJson(
        result["spatial_features"],
        style_function=lambda x: {'fillColor': '#ff0000' if x['properties']['severity'] == 'HIGH' else '#ffa500', 'color': 'red', 'weight': 2, 'fillOpacity': 0.4},
        tooltip=folium.GeoJsonTooltip(fields=['hazard', 'severity'])
    ).add_to(m)

    # Overlay Evacuation Shelters & Egress Corridors
    for shelter in st.session_state.shelters:
        folium.Marker(
            location=[shelter["lat"], shelter["lon"]],
            popup=f"<b>{shelter['name']}</b><br>Type: {shelter['type']}<br>Capacity: {shelter['capacity']:,} people",
            tooltip=shelter["name"],
            icon=folium.Icon(color="green", icon="plus-sign")
        ).add_to(m)

    for corridor in st.session_state.corridors:
        folium.PolyLine(
            locations=corridor["route_coordinates"],
            color="#28a745",
            weight=3,
            dash_array="6, 8",
            tooltip=f"Evacuation Route to {corridor['destination']}"
        ).add_to(m)

st_folium(m, width=1100, height=500)

# --- COMMAND & CONTROL SECTION ---
if st.session_state.sos_data:
    st.markdown("---")
    st.subheader("📡 Automated SOS Routing & Relief Logistics")
    
    col_sos1, col_sos2 = st.columns([1, 2])
    
    with col_sos1:
        st.info("**Dispatched Agencies:**")
        for agency in st.session_state.sos_data["agencies_dispatched"]:
            st.markdown(f"- 🚒 {agency}")
        
        st.markdown("**Designated Safe Shelters Active:**")
        for s in st.session_state.shelters:
            st.markdown(f"- 🟢 **{s['name']}** ({s['capacity']:,} beds)")
            
        st.markdown("---")
        target_phone = st.text_input("Responder Phone Number (incl. country code, e.g., +919876543210):")
        
        if st.button("🔥 TRANSMIT LIVE SMS ALERT", type="primary"):
            if target_phone:
                with st.spinner("Broadcasting to cellular networks..."):
                    success = send_emergency_sms(target_phone, st.session_state.sos_data)
                    if success:
                        st.success(f"SOS transmitted successfully to {target_phone}!")
                    else:
                        st.error("Transmission failed. Check Twilio credentials in terminal.")
            else:
                st.warning("Please enter a target phone number first.")
            
    with col_sos2:
        st.warning(st.session_state.sos_data["sos_text"])