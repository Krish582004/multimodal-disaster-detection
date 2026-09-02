def generate_safe_shelters(bbox: list, disaster_type: str) -> list:
    """
    Identifies designated safe relief assembly points positioned outside the disaster perimeter.
    In production, this queries OpenStreetMap (OSM) Overpass API for schools, stadiums, and hospitals.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    buffer = 0.08  # Distance outside the hazard zone

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Place designated shelters in 4 cardinal directions outside the perimeter
    shelters = [
        {
            "name": "North Relief Camp & Trauma Center",
            "lat": round(max_lat + buffer, 4),
            "lon": round(center_lon, 4),
            "type": "Primary Medical Shelter",
            "capacity": 2500
        },
        {
            "name": "East High-Ground Assembly Zone",
            "lat": round(center_lat, 4),
            "lon": round(max_lon + buffer, 4),
            "type": "Community Evacuation Center",
            "capacity": 1800
        },
        {
            "name": "South Staging Area",
            "lat": round(min_lat - buffer, 4),
            "lon": round(center_lon, 4),
            "type": "Logistics & Supply Base",
            "capacity": 1200
        },
        {
            "name": "West Emergency Heliport & Camp",
            "lat": round(center_lat, 4),
            "lon": round(min_lon - buffer, 4),
            "type": "Airlift & Evac Site",
            "capacity": 900
        }
    ]
    return shelters

def generate_evacuation_corridors(bbox: list, shelters: list) -> list:
    """
    Draws perimeter-skirting egress vectors leading out of the affected zone toward shelters.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    corridors = []
    for shelter in shelters:
        corridors.append({
            "destination": shelter["name"],
            "route_coordinates": [
                [center_lat, center_lon],
                [shelter["lat"], shelter["lon"]]
            ]
        })
    return corridors