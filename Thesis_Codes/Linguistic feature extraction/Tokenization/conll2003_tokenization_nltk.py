from datasets import load_dataset
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download necessary NLTK resources
nltk.download('punkt')

# Load the CoNLL-2003 dataset
conll2003_dataset = load_dataset("conll2003", trust_remote_code=True)

# Define NLTK tokenization function
def nltk_tokenize(text):
    return word_tokenize(text)

# Function to calculate token-level metrics by comparing with gold-standard tokens
def calculate_token_level_metrics(pred_tokens, gold_tokens):
    true_positives = sum(1 for token in pred_tokens if token in gold_tokens)
    false_positives = len(pred_tokens) - true_positives
    false_negatives = len(gold_tokens) - true_positives

    # Token-Level Accuracy calculation
    token_level_accuracy = true_positives / len(gold_tokens) if gold_tokens else 0

    # Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return token_level_accuracy, precision, recall, f1

# Initialize lists to store metrics for all sentences
token_level_accuracies = []
precisions = []
recalls = []
f1_scores = []

# Measure memory and time for tokenization
start_time = time.time()
memory_before = memory_usage()[0]

# Process all samples in the dataset
for document in conll2003_dataset["train"]:
    # Extract the gold-standard tokens from the dataset
    gold_tokens = document["tokens"]

    # Concatenate tokens into a single string for NLTK tokenization
    text = " ".join(gold_tokens)

    # Tokenize the text using NLTK
    nltk_tokens = nltk_tokenize(text)

    # Calculate token-level accuracy, precision, recall, and F1 score
    token_level_accuracy, precision, recall, f1 = calculate_token_level_metrics(nltk_tokens, gold_tokens)

    # Append metrics
    token_level_accuracies.append(token_level_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Calculate average metrics
avg_token_level_accuracy = sum(token_level_accuracies) / len(token_level_accuracies)
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1 = sum(f1_scores) / len(f1_scores)

# Measure memory usage after tokenization
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before

# Calculate time taken for tokenization
end_time = time.time()
time_taken = end_time - start_time

# Print tokenization accuracy metrics
print("Tokenization Metrics using NLTK on CoNLL-2003 dataset:")
print(f"Average Token-Level Accuracy: {avg_token_level_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# Print performance metrics
print("\nTokenization Performance Metrics:")
print(f"Time taken for tokenization: {time_taken:.4f} seconds")
print(f"Memory used for tokenization: {memory_used:.4f} MB")
