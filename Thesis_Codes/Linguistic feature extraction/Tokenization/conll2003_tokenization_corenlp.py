import stanza
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download and initialize Stanza with tokenization
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize', use_gpu=False)

# Load the CoNLL-2003 dataset
conll2003_dataset = load_dataset("conll2003", trust_remote_code=True)

# Define CoreNLP tokenization function
def corenlp_tokenize(text):
    doc = nlp(text)
    return [word.text for sentence in doc.sentences for word in sentence.words]

# Function to calculate token-level metrics
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

# Initialize lists to store metrics
token_level_accuracies = []
precisions = []
recalls = []
f1_scores = []

# Measure memory and time for tokenization
start_time = time.time()
memory_before = memory_usage()[0]

# Process the first 100 samples for efficiency
for idx, document in enumerate(conll2003_dataset['train']):

    # Extract gold-standard tokens from the dataset
    gold_tokens = document["tokens"]

    # Concatenate tokens into a single string for CoreNLP tokenization
    text = " ".join(gold_tokens)

    # Tokenize using CoreNLP
    corenlp_tokens = corenlp_tokenize(text)

    # Calculate token-level accuracy, precision, recall, and F1 score
    token_level_accuracy, precision, recall, f1 = calculate_token_level_metrics(corenlp_tokens, gold_tokens)

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

# Measure memory usage and time
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before
time_taken = time.time() - start_time

# Print results
print("Tokenization Metrics using CoreNLP on CoNLL-2003 dataset :")
print(f"Average Token-Level Accuracy: {avg_token_level_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken for tokenization: {time_taken:.4f} seconds")
print(f"Memory used for tokenization: {memory_used:.4f} MB")
