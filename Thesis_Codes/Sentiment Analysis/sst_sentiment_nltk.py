from datasets import load_dataset
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
import re
from collections import Counter
import time
from memory_profiler import memory_usage

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load the SST dataset (default configuration)
dataset = load_dataset("sst", "default")
train_data = dataset["train"]
test_data = dataset["test"]

# Map float labels to binary sentiment classes
# 0.0–0.4 → Negative (0), 0.5–1.0 → Positive (1)
def map_sst_labels(label):
    return 0 if label < 0.5 else 1

# Apply the mapping to create binary labels
test_data = test_data.map(lambda x: {"binary_label": map_sst_labels(x["label"])})

# Preprocess text to normalize it
def preprocess_text(text):
    text = re.sub(r"http\S+", "link", text)  # Replace URLs
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Define a function to classify sentiment using NLTK's SentimentIntensityAnalyzer
def classify_sentiment_nltk(text):
    text = preprocess_text(text)
    scores = sia.polarity_scores(text)
    return 1 if scores['compound'] >= 0.05 else 0  # Adjusted thresholds

# Initialize storage for predictions and true labels
true_labels = []
predicted_labels = []

# Measure memory and time
start_time = time.time()
memory_before = memory_usage()[0]

# Evaluate on the test dataset
for example in test_data:
    text = example["sentence"]
    true_label = example["binary_label"]

    # Classify sentiment using NLTK
    predicted_label = classify_sentiment_nltk(text)

    # Append results
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

# Measure memory and time
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before
time_taken = time.time() - start_time

# Calculate performance metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="binary", zero_division=0)
recall = recall_score(true_labels, predicted_labels, average="binary", zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average="binary", zero_division=0)

# Print label distributions for debugging
print("True Label Distribution:", Counter(true_labels))
print("Predicted Label Distribution:", Counter(predicted_labels))

# Print results
print("Sentiment Analysis Metrics using NLTK on SST Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken for sentiment analysis: {time_taken:.4f} seconds")
print(f"Memory used for sentiment analysis: {memory_used:.4f} MB")
