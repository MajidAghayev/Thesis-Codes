from stanza.server import CoreNLPClient, StartServer
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Load the correct Amazon Reviews 2023 dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Magazine_Subscriptions", trust_remote_code=True)
data = dataset["full"].to_pandas()


# Define mapping from rating to sentiment
def map_rating_to_sentiment(rating):
    if rating >= 4.0:
        return "positive"
    elif rating <= 2.0:
        return "negative"
    else:
        return "neutral"

# Add a gold standard sentiment column based on rating
data["gold_sentiment"] = data["rating"].apply(map_rating_to_sentiment)

# Function to perform sentiment analysis using CoreNLP
CORENLP_SERVER_URL = "http://localhost:9000"

def corenlp_sentiment(text):
    with CoreNLPClient(annotators=['sentiment'], endpoint=CORENLP_SERVER_URL, timeout=30000, start_server=StartServer) as client:
        annotation = client.annotate(text)
        sentiments = [sentence.sentiment for sentence in annotation.sentence]
        sentiment_label = max(set(sentiments), key=sentiments.count)
        return sentiment_label.lower()

# Function to calculate sentiment analysis metrics
def calculate_sentiment_metrics(pred_sentiments, gold_sentiments):
    accuracy = accuracy_score(gold_sentiments, pred_sentiments)
    precision = precision_score(gold_sentiments, pred_sentiments, average='weighted')
    recall = recall_score(gold_sentiments, pred_sentiments, average='weighted')
    f1 = f1_score(gold_sentiments, pred_sentiments, average='weighted')
    return accuracy, precision, recall, f1

# Initialize metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Measure memory and time
start_time = time.time()
memory_before = memory_usage()[0]

# Evaluate on the dataset
pred_sentiments = []
gold_sentiments = data["gold_sentiment"].tolist()

for idx, row in data.iterrows():
    text = row["text"][:1000]  # Truncate text to 1000 characters
    pred_sentiment = corenlp_sentiment(text)
    pred_sentiments.append(pred_sentiment)

# Calculate metrics
accuracy, precision, recall, f1 = calculate_sentiment_metrics(pred_sentiments, gold_sentiments)
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)

# Calculate average metrics
avg_accuracy = sum(accuracies) / len(accuracies)
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1 = sum(f1_scores) / len(f1_scores)

# Measure memory usage and time
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before
time_taken = time.time() - start_time

# Print metrics
print("Sentiment Analysis Metrics using CoreNLP on Amazon Reviews 2023 dataset with Gold Standard:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken for sentiment analysis: {time_taken:.4f} seconds")
print(f"Memory used for sentiment analysis: {memory_used:.4f} MB")
