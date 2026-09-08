"""
src/fusion.py
Multimodal Fusion Engine: Combines Satellite Vision AI scores with GDACS NLP Context scores.
"""

import sys
import os

# Ensure local src imports work seamlessly regardless of run context
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import generate_multimodal_tensor
from model import run_vision_inference
from nlp_module import fetch_live_global_disasters


def run_multimodal_fusion(disaster_event: dict) -> dict:
    """
    Synthesizes visual satellite inference and text context scores into a single threat metric.
    """
    bbox = disaster_event.get("bbox", [0, 0, 0, 0])
    nlp_score = disaster_event.get("nlp_score", 0.50)
    
    # 1. Trigger ECE Spatial AI Pipeline
    spatial_tensor = generate_multimodal_tensor(bbox)
    vision_score = run_vision_inference(spatial_tensor)
    
    # 2. Weighted Multimodal Fusion Calculation (65% Vision / 35% NLP)
    fused_score = round((0.65 * vision_score) + (0.35 * nlp_score), 3)
    
    # 3. Categorize Threat Level
    if fused_score >= 0.75:
        severity = "HIGH"
    elif fused_score >= 0.50:
        severity = "MEDIUM"
    else:
        severity = "LOW"
        
    # 4. Construct Spatial Feature for Folium rendering
    spatial_feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]]
                    ]]
                },
                "properties": {
                    "hazard": disaster_event.get("disaster_type", "unknown"),
                    "severity": severity,
                    "confidence": f"{fused_score * 100:.1f}%"
                }
            }
        ]
    }
    
    return {
        "event_id": disaster_event.get("id"),
        "event_name": disaster_event.get("name"),
        "hazard_type": disaster_event.get("disaster_type"),
        "vision_score": vision_score,
        "nlp_score": nlp_score,
        "fused_score": fused_score,
        "severity": severity,
        "spatial_features": spatial_feature
    }


if __name__ == "__main__":
    live_events = fetch_live_global_disasters()
    if live_events:
        assessment = run_multimodal_fusion(live_events[0])
        print("Fusion test success:", assessment["fused_score"])