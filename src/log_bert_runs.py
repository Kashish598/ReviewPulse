import mlflow


def log_candidate(run_name, checkpoint, mapping, acc, f1, is_selected=False):
    with mlflow.start_run(run_name=run_name):
        # Log parameters[cite: 4]
        mlflow.log_param("checkpoint", checkpoint)
        mlflow.log_param("label_mapping", mapping)
        mlflow.log_param("max_token_length", 512)
        mlflow.log_param("fine_tuned", False)
        mlflow.log_param("selected_for_api", is_selected)

        # Log metrics[cite: 4]
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_macro_f1", f1)

        print(f"Logged {run_name} to MLflow.")


def main():
    mlflow.set_experiment("reviewpulse-bert")

    # Run 1: DistilBERT (Candidate)
    log_candidate(
        run_name="eval_distilbert",
        checkpoint="distilbert-base-uncased-finetuned-sst-2-english",
        mapping="Negative -> 0, Positive -> 1",
        acc=0.891,
        f1=0.8909,
    )

    # Run 2: RoBERTa (Candidate)
    log_candidate(
        run_name="eval_roberta",
        checkpoint="cardiffnlp/twitter-roberta-base-sentiment",
        mapping="Negative/Neutral -> 0, Positive -> 1",
        acc=0.897,
        f1=0.8969,
    )

    # Run 3: nlptown (Winner)
    log_candidate(
        run_name="eval_nlptown_winner",
        checkpoint="nlptown/bert-base-multilingual-uncased-sentiment",
        mapping="1-3 stars -> 0, 4-5 stars -> 1",
        acc=0.930,
        f1=0.9298,
        is_selected=True,
    )


if __name__ == "__main__":
    main()
