import pandas as pd
from pathlib import Path
from src import bert_model

def main():
    # 1. Load the test subset (using the fixture for development)
    test_file = Path("data/sample/test.csv")
    if not test_file.exists():
        raise FileNotFoundError(f"Could not find {test_file}. Did you run prepare_data --fixture?")
    
    test_df = pd.read_csv(test_file)
    
    # 2. Map the true labels (0/1) to the shared string contract ('negative'/'positive')[cite: 2]
    label_map = {0: "negative", 1: "positive"}
    test_df["label_true"] = test_df["label"].map(label_map)
    
    # 3. Load the locally saved model[cite: 3, 4]
    print("Loading local BERT model...")
    bert_model.load_model("models/nlptown_sentiment")
    
    # 4. Generate predictions[cite: 2]
    print(f"Running predictions on {len(test_df)} test reviews...")
    predictions = bert_model.predict(test_df["review_text"].tolist())
    
    # 5. Format the output to match the shared contract exactly[cite: 2]
    results_df = pd.DataFrame({
        "row_id": test_df["row_id"],
        "label_true": test_df["label_true"],
        "pred": [p["label"] for p in predictions],
        "score": [p["score"] for p in predictions]
    })
    
    # 6. Save the predictions file
    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bert_test.csv"
    results_df.to_csv(out_path, index=False)
    
    print(f"Success! Predictions saved to {out_path}")

if __name__ == "__main__":
    main()