"""
src/nlp_module.py
IT Track: Live global disaster feed ingestion (GDACS GeoJSON) and NLP context scoring.
"""

import requests


def fetch_live_global_disasters(event_types: list = ["FL", "WF", "TC", "DR", "EQ"]) -> list:
    """
    Fetches real-time active disaster alerts from the official UN/EC GDACS GeoJSON feed.
    """
    url = "https://www.gdacs.org/xml/gdacs.geojson"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events_data = response.json()
    except Exception as e:
        print(f"Failed to fetch live GDACS alerts: {e}")
        return []

    active_disasters = []

    features = events_data.get("features", [])
    for feature in features:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        
        event_type = props.get("eventtype")
        if event_type in event_types:
            coords = geometry.get("coordinates", [])
            
            # Safely unwrap nested coordinate lists (Polygons / MultiPolygons)
            while isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list):
                coords = coords[0]

            # Skip entries without valid [lon, lat] pairs
            if not isinstance(coords, list) or len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]
            
            bbox = [
                round(lon - 0.15, 4),
                round(lat - 0.15, 4),
                round(lon + 0.15, 4),
                round(lat + 0.15, 4)
            ]

            alert_level = str(props.get("alertlevel", "Green")).upper()
            title = props.get("name", props.get("eventname", "Unknown Event"))
            country = props.get("country", "Global Region")
            
            # Explicitly map the UN GDACS acronyms to their full string names
            hazard_map = {
                "FL": "flood",
                "WF": "wildfire",
                "TC": "cyclone",
                "DR": "drought",
                "EQ": "earthquake"
            }
            mapped_hazard = hazard_map.get(event_type, "unknown")
            
            score_map = {"RED": 0.95, "ORANGE": 0.75, "GREEN": 0.40}
            nlp_score = score_map.get(alert_level, 0.50)

            active_disasters.append({
                "id": props.get("eventid"),
                "name": f"{title} ({country})",
                "disaster_type": mapped_hazard,
                "alert_level": alert_level,
                "bbox": bbox,
                "lat": lat,
                "lon": lon,
                "nlp_score": nlp_score,
                "description": props.get("description", "No detailed summary available.")
            })

    return active_disasters


if __name__ == "__main__":
    print("--- LIVE GLOBAL DISASTER SCANNER ---")
    live_events = fetch_live_global_disasters()
    
    print(f"Found {len(live_events)} active disaster alerts worldwide.\n")
    for event in live_events[:5]:
        print(f"🚨 [{event['alert_level']}] {event['name']}")
        print(f"   Hazard Type: {event['disaster_type'].upper()}")
        print(f"   Bounding Box: {event['bbox']}")
        print(f"   NLP Risk Score: {event['nlp_score']}")
        print("-" * 50)