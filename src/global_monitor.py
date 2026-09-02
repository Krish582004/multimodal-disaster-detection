import time
import urllib.request
import json
from datetime import datetime
from src.pipeline import search_satellite_data
from src.preprocess import generate_multimodal_tensor
from src.model import run_vision_inference
from src.nlp_module import get_text_disaster_score
from src.fusion import generate_final_payload
from src.sos_router import generate_sos_payload

def fetch_live_global_hazards():
    """Pulls live, real-world hazard data from the US Geological Survey (USGS)."""
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("features", [])
    except Exception as e:
        print(f"Global satellite telemetry error: {e}")
        return []

def coordinate_to_bbox(lon: float, lat: float, buffer: float = 0.15) -> list:
    """Converts a single global pinpoint into a scanning box for the AI."""
    return [lon - buffer, lat - buffer, lon + buffer, lat + buffer]

def start_global_scan(scan_interval_seconds=60):
    print(f"🌍 Initiating Global Monitoring Daemon... [Scanning every {scan_interval_seconds}s]")
    processed_event_ids = set()
    
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sweeping global telemetry feeds...")
        hazards = fetch_live_global_hazards()
        new_hazards = [h for h in hazards if h["id"] not in processed_event_ids]
        
        if not new_hazards:
            print("No new significant global anomalies detected. Standing by.")
        
        for hazard in new_hazards:
            event_id = hazard["id"]
            props = hazard["properties"]
            coords = hazard["geometry"]["coordinates"]
            
            lon, lat = coords[0], coords[1]
            bbox = coordinate_to_bbox(lon, lat)
            disaster_type = "earthquake" 
            
            print(f"\n⚠️ ANOMALY DETECTED: {props['title']}")
            print(f"📍 Engaging target sector: {bbox}")
            print("🧠 Initializing Multimodal AI Validation...")
            
            try:
                search_satellite_data(bbox, days_back=30)
                tensor = generate_multimodal_tensor(bbox)
                vis_score = run_vision_inference(tensor)
                nlp_score = get_text_disaster_score(bbox, target_hazard=disaster_type)
                result = generate_final_payload(bbox, disaster_type, vis_score, nlp_score)
                
                if result["metrics"]["alert_triggered"]:
                    print("🚨 AI VALIDATION CONFIRMED. GENERATING SOS ROUTING PROTOCOL.")
                    sos_data = generate_sos_payload(
                        disaster_type=disaster_type,
                        severity=result["spatial_features"]["features"][0]["properties"]["severity"],
                        bbox=bbox,
                        timestamp=result["timestamp"]
                    )
                    print("\n--- AUTOMATED SOS DISPATCH ---")
                    print(sos_data["sos_text"])
                    print(f"Transmitting to: {', '.join(sos_data['agencies_dispatched'])}")
                    print("------------------------------")
                else:
                    print("✅ AI determined hazard severity is low. No SOS required.")
                    
            except Exception as e:
                print(f"System failure during AI validation: {e}")
            
            processed_event_ids.add(event_id)
            time.sleep(5) 
            
        time.sleep(scan_interval_seconds)

if __name__ == "__main__":
    start_global_scan()