from pystac_client import Client
import datetime

def search_satellite_data(bbox: list, days_back: int = 7):
    """
    Searches for Sentinel-1 (Radar) and Sentinel-2 (Optical) imagery.
    
    Args:
        bbox (list): [min_lon, min_lat, max_lon, max_lat]
        days_back (int): How many days into the past to search
        
    Returns:
        dict: A dictionary containing STAC items for S1 and S2.
    """
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    # Define time range
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=days_back)
    time_range = f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    print(f"Connecting to {catalog_url}...")
    client = Client.open(catalog_url)

    # Search Sentinel-2 (Optical)
    print(f"Searching Sentinel-2 for bbox: {bbox}")
    s2_search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 100}} # Less than 30% cloud cover
    )
    s2_items = list(s2_search.items())

    # Search Sentinel-1 (Radar/SAR)
    print(f"Searching Sentinel-1 for bbox: {bbox}")
    s1_search = client.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=time_range
    )
    s1_items = list(s1_search.items())

    return {
        "sentinel_2_optical": s2_items,
        "sentinel_1_radar": s1_items
    }

if __name__ == "__main__":
    # Test the pipeline with Kolkata bbox
    sample_bbox = [88.20, 22.45, 88.50, 22.70]
    results = search_satellite_data(sample_bbox, days_back=30)
    
    print(f"\n--- API SEARCH RESULTS ---")
    print(f"Found {len(results['sentinel_2_optical'])} Optical images.")
    print(f"Found {len(results['sentinel_1_radar'])} Radar images.")
    
    if results['sentinel_2_optical']:
        print("\nSample Optical Asset URLs (Cloud Optimized GeoTIFFs):")
        item = results['sentinel_2_optical'][0]
        print(f"Red Band (B04): {item.assets['B04'].href}")
        print(f"Visual (TCI): {item.assets['visual'].href}")