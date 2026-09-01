from transformers import pipeline
import time

def fetch_local_alerts(bbox: list) -> str:
    """
    Simulates scraping live news APIs or Twitter/X feeds for a specific bounding box.
    In production, you would connect to the GDELT Project API or X API here.
    """
    print(f"Scraping social and news APIs for bounding box: {bbox}...")
    time.sleep(1) # Simulate network delay
    
    # Simulated live text feeds from the region
    mock_live_feed = (
        "Severe waterlogging reported in the city center. "
        "Authorities are issuing evacuation warnings for riverbank areas. "
        "Traffic is at a standstill due to monsoon flooding."
    )
    return mock_live_feed

def get_text_disaster_score(bbox: list, target_hazard: str = "flood") -> float:
    """
    Analyzes local text feeds using a Hugging Face Large Language Model
    to determine the probability of an active disaster.
    """
    live_text = fetch_local_alerts(bbox)
    print("Loading Hugging Face NLP model (this may take a moment to download the first time)...")
    
    # Initialize a Zero-Shot Classifier
    # We use a lightweight model (valhalla/distilbart-mnli-12-3) for faster local execution
    classifier = pipeline(
        "zero-shot-classification", 
        model="valhalla/distilbart-mnli-12-3"
    )
    
    # Define the categories we want the AI to sort the text into
    labels = [f"active {target_hazard} disaster", "normal weather", "safe and calm"]
    
    print("Analyzing semantics and context...")
    result = classifier(live_text, candidate_labels=labels)
    
    # Extract the probability score for the disaster label
    disaster_label = f"active {target_hazard} disaster"
    disaster_index = result["labels"].index(disaster_label)
    confidence_score = result["scores"][disaster_index]
    
    return round(confidence_score, 3)

if __name__ == "__main__":
    print("--- IT NLP CONTEXT TEST ---")
    sample_bbox = [88.20, 22.45, 88.50, 22.70]
    
    try:
        score = get_text_disaster_score(sample_bbox, target_hazard="flood")
        print("\n✅ Text Analysis Complete!")
        print(f"NLP Disaster Confidence Score: {score * 100:.1f}%")
        
        if score > 0.60:
            print("Context validates visual data: High likelihood of real emergency.")
        else:
            print("Context implies false positive: Likely just routine weather.")
            
    except Exception as e:
        print(f"Error during NLP processing: {e}")