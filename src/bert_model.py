import os
from transformers import pipeline

# Global pipeline instance
_classifier = None

def load_model(model_path: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Initializes the sentiment pipeline. 
    By default it uses the Hugging Face repo, but can point to a local saved directory.
    """
    global _classifier
    _classifier = pipeline(
        "sentiment-analysis", 
        model=model_path, 
        tokenizer=model_path,
        truncation=True, 
        max_length=512
    )

def map_stars_to_label(star_label: str) -> str:
    """
    Applies the explicit Star Mapping Policy:
    1-3 stars -> negative
    4-5 stars -> positive
    """
    if star_label in ["1 star", "2 stars", "3 stars"]:
        return "negative"
    elif star_label in ["4 stars", "5 stars"]:
        return "positive"
    else:
        raise ValueError(f"Unexpected label format: {star_label}")

def predict(texts: list[str]) -> list[dict]:
    """
    Predictor interface.
    Returns a list of dictionaries with the mapped 'negative'/'positive' label 
    and the model's confidence score.
    """
    if _classifier is None:
        # Fallback initialization if not explicitly loaded
        load_model()
        
    raw_predictions = _classifier(texts)
    
    results = []
    for pred in raw_predictions:
        mapped_label = map_stars_to_label(pred["label"])
        results.append({
            "label": mapped_label,
            "score": pred["score"]  # Including optional scores as supported
        })
        
    return results