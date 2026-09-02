from datetime import datetime

AGENCY_MAP = {
    "flood": ["NDRF", "SDRF", "District Disaster Management Authority"],
    "wildfire": ["Fire & Emergency Services", "Forest Department", "Police"],
    "cyclone": ["District Disaster Management Authority", "NDRF", "Coast Guard"],
    "earthquake": ["NDRF", "Fire & Emergency Services", "Police", "Medical Responders"],
    "landslide": ["District Administration", "SDRF", "Local Rescue Teams"]
}

def calculate_area_km2(bbox: list) -> float:
    width_km = abs(bbox[2] - bbox[0]) * 111
    height_km = abs(bbox[3] - bbox[1]) * 111
    return round(width_km * height_km, 2)

def generate_sos_payload(disaster_type: str, severity: str, bbox: list, timestamp: str) -> dict:
    area_km2 = calculate_area_km2(bbox)
    population_density = 4500 if "flood" in disaster_type else 800
    impact_multiplier = 0.85 if severity == "HIGH" else 0.30
    
    est_people = int(area_km2 * population_density * impact_multiplier)
    est_animals = int(est_people * 0.2)
    agencies = AGENCY_MAP.get(disaster_type.lower(), ["General Emergency Services"])
    
    sos_message = f"""🚨 **CRITICAL DISASTER ALERT** 🚨
**Disaster:** {disaster_type.upper()}
**Location (BBox):** {bbox}
**Severity:** {severity}
**Estimated Affected Area:** {area_km2} km²
**Estimated People at Risk:** {est_people:,}
**Animals/Livestock at Risk:** {est_animals:,}
**Detected at:** {timestamp}
**Recommended Response:** Immediate evacuation and perimeter control."""

    return {
        "agencies_dispatched": agencies,
        "impact_metrics": {"area_km2": area_km2, "est_people": est_people, "est_animals": est_animals},
        "sos_text": sos_message
    }