import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import mlflow

def main():
    # 1. Configuration
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    save_dir = Path("models/nlptown_sentiment")
    
    # 2. Download the winning model and tokenizer
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # 3. Save the artifact locally
    print(f"Saving artifact to {save_dir}...")
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    # 4. Save the required metadata file
    metadata = {
        "model_name": model_name,
        "label_mapping": "1-3 stars -> negative, 4-5 stars -> positive",
        "max_token_length": 512,
        "fine_tuned": False
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # 5. Log the run to MLflow[cite: 4]
    mlflow.set_experiment("reviewpulse-bert")  #[cite: 3]
    with mlflow.start_run(run_name="save_best_pretrained"):
        mlflow.log_params(metadata)
        mlflow.log_artifact(str(save_dir / "metadata.json"))
        
    print(f"Done! Model safely stored in {save_dir}/ and kept out of Git.")

if __name__ == "__main__":
    main()