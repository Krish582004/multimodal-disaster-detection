"""
app.py
Main Streamlit Dashboard for Omni-Hazard Multimodal Disaster Detection System.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium

from src.nlp_module import fetch_live_global_disasters
from src.fusion import run_multimodal_fusion
from src.alert_system import dispatch_sos

st.set_page_config(
    page_title="Omni-Hazard Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🚨 Omni-Hazard Multimodal Disaster Detection System")
st.markdown("Fusing UN GDACS NLP event feeds with Sentinel-1/2 Vision AI models for real-time hazard evaluation.")

# --- SIDEBAR DATA FETCHING ---
st.sidebar.header("Global Disaster Stream")

@st.cache_data(ttl=300)
def load_gdacs_events():
    return fetch_live_global_disasters()

live_events = load_gdacs_events()

if not live_events:
    st.sidebar.error("Unable to load live GDACS alerts. Check network connection.")
    st.stop()

event_options = {f"[{e['alert_level']}] {e['name']}": e for e in live_events}
selected_label = st.sidebar.selectbox("Select Active Alert", list(event_options.keys()))
selected_event = event_options[selected_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Selected Event Details")
st.sidebar.write(f"**Hazard:** {selected_event['disaster_type'].upper()}")
st.sidebar.write(f"**UN Level:** {selected_event['alert_level']}")
st.sidebar.write(f"**Coordinates:** [{selected_event['lat']}, {selected_event['lon']}]")
st.sidebar.write(f"**BBox:** {selected_event['bbox']}")

# --- SESSION STATE MANAGEMENT ---
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "analyzed_event_id" not in st.session_state:
    st.session_state.analyzed_event_id = None

# Automatically clear old results if the user selects a new disaster from the sidebar
if st.session_state.analyzed_event_id != selected_event["id"]:
    st.session_state.assessment = None

# --- MAIN DASHBOARD ---
st.write(f"### Analyzing Target: **{selected_event['name']}**")

if st.button("Run Multimodal Fusion Analysis", type="primary"):
    with st.spinner("Fetching spatial tensors & synthesizing neural scores..."):
        st.session_state.assessment = run_multimodal_fusion(selected_event)
        st.session_state.analyzed_event_id = selected_event["id"]

# Render results persistently whenever they exist in session state for the active event
if st.session_state.assessment is not None and st.session_state.analyzed_event_id == selected_event["id"]:
    assessment = st.session_state.assessment

    st.success("Multimodal Inference Complete!")

    # Display Threat Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Vision AI Score (65%)", f"{assessment['vision_score'] * 100:.1f}%")
    col2.metric("NLP Context Score (35%)", f"{assessment['nlp_score'] * 100:.1f}%")
    col3.metric("Final Fused Severity", f"{assessment['fused_score'] * 100:.1f}%", delta=f"Threat: {assessment['severity']}")

    # --- PROFESSIONAL UI TABS ---
    st.markdown("---")
    tab1, tab2 = st.tabs(["🗺️ Spatial Threat Map", "🚨 Emergency Dispatch Protocol"])

    with tab1:
        st.subheader("Geographical Footprint & Bounding Box Overlay")

        # Initialize Folium Map centered on hazard epicenter
        m = folium.Map(location=[selected_event["lat"], selected_event["lon"]], zoom_start=9)

        # Add Event Epicenter Marker
        folium.Marker(
            [selected_event["lat"], selected_event["lon"]],
            popup=selected_event["name"],
            tooltip=selected_event["disaster_type"].upper(),
            icon=folium.Icon(color="red" if selected_event["alert_level"] == "RED" else "orange")
        ).add_to(m)

        # Render Bounding Box Polygon from Fusion Output
        folium.GeoJson(
            assessment["spatial_features"],
            style_function=lambda x: {
                "fillColor": "#ff0000" if assessment["severity"] == "HIGH" else "#ffa500",
                "color": "#ff0000" if assessment["severity"] == "HIGH" else "#ffa500",
                "weight": 2,
                "fillOpacity": 0.35
            }
        ).add_to(m)

        st_folium(m, width=1000, height=500)

    with tab2:
        st.subheader("Automated SOS Dispatch")
        st.write(f"Current Hazard: **{selected_event['disaster_type'].upper()}** | Calculated Severity: **{assessment['severity']}**")
        
        # Render the SOS button unconditionally inside the dispatch tab
        if st.button("Broadcast SOS to Local Authorities", type="primary"):
            with st.spinner("Dispatching emergency payload..."):
                dispatch_msg = dispatch_sos(
                    event_name=selected_event["name"],
                    hazard_type=selected_event["disaster_type"],
                    severity=assessment["severity"],
                    lat=selected_event["lat"],
                    lon=selected_event["lon"]
                )
            st.success("SOS Alert Successfully Broadcasted via Secure SMTP!")
            st.code(dispatch_msg, language="text")