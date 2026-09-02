import time
import urllib.request
import json
from datetime import datetime, timezone
from src.pipeline import search_satellite_data
from src.preprocess import generate_multimodal_tensor
from src.model import run_vision_inference
from src.nlp_module import get_text_disaster_score
from src.fusion import generate_final_payload
from src.sos_router import generate_sos_payload


def fetch_live_global_hazards(enable_test_fallback: bool = True):
    """
    Pulls live hazard data from USGS. Injects a test event if no active events exist.
    """
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
    hazards = []
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            hazards = data.get("features", [])
    except Exception as e:
        print(f"Global telemetry notice: {e}")

    # If the global feed has no major events right now, inject a synthetic test event
    if not hazards and enable_test_fallback:
        print("ℹ️ No active significant events on live feed. Injecting synthetic test event...")
        hazards = [
            {
                "id": "test_event_live_001",
                "properties": {
                    "title": "M 6.4 - Test Seismic Anomaly (Simulated Trigger)",
                    "mag": 6.4,
                },
                "geometry": {
                    "coordinates": [88.36, 22.57, 10.0]  # Centered on Kolkata coordinates
                },
            }
        ]

    return hazards


def coordinate_to_bbox(lon: float, lat: float, buffer: float = 0.15) -> list:
    """Converts a single global pinpoint into a scanning box for the AI."""
    return [round(lon - buffer, 4), round(lat - buffer, 4), round(lon + buffer, 4), round(lat + buffer, 4)]


def start_global_scan(scan_interval_seconds=30):
    print(f"🌍 Initiating Global Monitoring Daemon... [Interval: {scan_interval_seconds}s]")
    processed_event_ids = set()

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sweeping global telemetry feeds...")
        hazards = fetch_live_global_hazards(enable_test_fallback=True)
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
            print(f"📍 Engaging target sector bbox: {bbox}")
            print("🧠 Initializing Multimodal AI Validation...")

            try:
                # 1. Satellite telemetry query
                search_satellite_data(bbox, days_back=30)

                # 2. Sensor tensor stacking
                tensor = generate_multimodal_tensor(bbox)

                # 3. Vision AI inference
                vis_score = run_vision_inference(tensor)

                # 4. Context NLP scoring
                nlp_score = get_text_disaster_score(bbox, target_hazard=disaster_type)

                # 5. Fusion calculation
                result = generate_final_payload(bbox, disaster_type, vis_score, nlp_score)

                # 6. Automated SOS routing
                if result["metrics"]["alert_triggered"]:
                    print("\n🚨 AI VALIDATION CONFIRMED: HAZARD LEVEL CRITICAL")
                    sos_data = generate_sos_payload(
                        disaster_type=disaster_type,
                        severity=result["spatial_features"]["features"][0]["properties"]["severity"],
                        bbox=bbox,
                        timestamp=result["timestamp"]
                    )
                    print("\n" + "=" * 45)
                    print("📡 TRANSMITTING AUTOMATED EMERGENCY SOS")
                    print("=" * 45)
                    print(sos_data["sos_text"])
                    print(f"\nDispatched Authorities: {', '.join(sos_data['agencies_dispatched'])}")
                    print("=" * 45 + "\n")
                else:
                    print("✅ AI determined hazard severity is below alert threshold. No SOS triggered.")

            except Exception as e:
                print(f"Pipeline error during AI verification: {e}")

            processed_event_ids.add(event_id)
            time.sleep(2)

        print(f"Cycle finished. Sleeping for {scan_interval_seconds} seconds...")
        time.sleep(scan_interval_seconds)


if __name__ == "__main__":
    start_global_scan()