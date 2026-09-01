from datetime import datetime, timezone

def generate_final_payload(bbox: list, disaster_type: str, vis_conf: float, nlp_conf: float) -> dict:
    # 70% weight to visual sensors, 30% to text context
    fused_score = round((vis_conf * 0.70) + (nlp_conf * 0.30), 3)
    
    # Generate a localized polygon for the map visualization
    d_lon, d_lat = (bbox[2] - bbox[0]) * 0.25, (bbox[3] - bbox[1]) * 0.25
    
    return {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disaster_type": disaster_type,
        "metrics": {
            "visual_confidence": vis_conf,
            "nlp_confidence": nlp_conf,
            "fused_score": fused_score,
            "alert_triggered": fused_score >= 0.70,
        },
        "spatial_features": {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"hazard": disaster_type, "severity": "HIGH" if fused_score > 0.8 else "MEDIUM"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[bbox[0]+d_lon, bbox[1]+d_lat], [bbox[2]-d_lon, bbox[1]+d_lat], 
                                     [bbox[2]-d_lon, bbox[3]-d_lat], [bbox[0]+d_lon, bbox[3]-d_lat], 
                                     [bbox[0]+d_lon, bbox[1]+d_lat]]]
                }
            }]
        }
    }