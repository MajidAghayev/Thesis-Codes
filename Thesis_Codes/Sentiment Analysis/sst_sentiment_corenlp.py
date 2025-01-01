import time
import psutil
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stanza.server import CoreNLPClient

# Load the SST dataset from Hugging Face
dataset = load_dataset("sst", split="train")


# Convert the floating-point labels to binary labels (threshold: 0.5)
def label_to_binary(label, threshold=0.5):
    return 1 if label >= threshold else 0


# Preprocess the dataset to include binary labels
dataset = dataset.map(lambda example: {"binary_label": label_to_binary(example["label"]), "text": example["sentence"]})


# Clean input text
def clean_text(text):
    """Remove unnecessary characters and normalize the text."""
    import re
    text = re.sub(r"[^\w\s.,!?\'\"-]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


# Initialize CoreNLP Client
with CoreNLPClient(
        annotators=["sentiment"],
        timeout=15000,
        memory="4G",
) as client:
    # Function to predict sentiment using CoreNLP
    def predict_sentiment(text):
        cleaned_text = clean_text(text)

        # Annotate the text
        ann = client.annotate(cleaned_text)

        # Extract sentiment for the first sentence
        try:
            sentiment = ann.sentence[0].sentiment  # e.g., "VeryPositive", "Positive", etc.

            # Map sentiment labels to binary values
            sentiment_map = {
                "VeryNegative": 0,
                "Negative": 0,
                "Neutral": 0,  # Treat Neutral as Negative
                "Positive": 1,
                "VeryPositive": 1,
            }
            return sentiment_map.get(sentiment, 0)  # Default to Negative if unexpected label
        except IndexError:
            return 0  # Default to Negative if no sentences found


    # Measure time and memory usage before analysis
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Apply CoreNLP sentiment analysis

    dataset = dataset.map(lambda example: {"predicted_label": predict_sentiment(example["text"])})

    # Measure time and memory usage after analysis
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    # Calculate evaluation metrics
    y_true = dataset["binary_label"]
    y_pred = dataset["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Time taken for sentiment analysis: {time_taken:.4f} seconds")
    print(f"Memory used for sentiment analysis: {memory_used:.4f} MB")
