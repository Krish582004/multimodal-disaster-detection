from datetime import datetime, timezone
import random


def get_mock_prediction(bbox: list, disaster_type: str = "flood") -> dict:
    """
    Simulates the fused output of the spatial AI model and NLP text classifier.

    Args:
        bbox (list): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
        disaster_type (str): "flood", "wildfire", or "landslide"

    Returns:
        dict: Standardized disaster payload formatted for GeoJSON/Folium mapping.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 coordinates: [min_lon, min_lat, max_lon, max_lat]")

    min_lon, min_lat, max_lon, max_lat = bbox

    # Generate synthetic confidence values
    visual_conf = round(random.uniform(0.75, 0.96), 3)
    nlp_conf = round(random.uniform(0.60, 0.90), 3)
    fused_score = round((visual_conf * 0.70) + (nlp_conf * 0.30), 3)

    # Subdivide bounding box to simulate a localized affected hazard zone
    delta_lon = (max_lon - min_lon) * 0.25
    delta_lat = (max_lat - min_lat) * 0.25

    zone_min_lon = min_lon + delta_lon
    zone_max_lon = max_lon - delta_lon
    zone_min_lat = min_lat + delta_lat
    zone_max_lat = max_lat - delta_lat

    payload = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disaster_type": disaster_type.lower(),
        "bounding_box": bbox,
        "metrics": {
            "visual_confidence": visual_conf,
            "nlp_confidence": nlp_conf,
            "fused_score": fused_score,
            "alert_triggered": fused_score >= 0.70,
        },
        "spatial_features": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "hazard": disaster_type.lower(),
                        "severity": "HIGH" if fused_score > 0.80 else "MEDIUM",
                        "confidence": fused_score,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [zone_min_lon, zone_min_lat],
                                [zone_max_lon, zone_min_lat],
                                [zone_max_lon, zone_max_lat],
                                [zone_min_lon, zone_max_lat],
                                [zone_min_lon, zone_min_lat],
                            ]
                        ],
                    },
                }
            ],
        },
    }

    return payload


if __name__ == "__main__":
    # Test execution with a sample bounding box (Kolkata region)
    sample_bbox = [88.20, 22.45, 88.50, 22.70]
    result = get_mock_prediction(sample_bbox, disaster_type="flood")
    
    print("--- MOCK DATA CONTRACT TEST ---")
    print(f"Status: {result['status']}")
    print(f"Hazard: {result['disaster_type'].upper()}")
    print(f"Fused Alert Score: {result['metrics']['fused_score']}")
    print(f"Alert Triggered: {result['metrics']['alert_triggered']}")
    print(f"Polygon Geometry Generated: {len(result['spatial_features']['features'])} feature(s)")