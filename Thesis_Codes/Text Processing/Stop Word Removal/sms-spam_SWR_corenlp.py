from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stanza.server import CoreNLPClient, StartServer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import nltk
import time

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the SMS Spam dataset
dataset = load_dataset("ucirvine/sms_spam", trust_remote_code=True)
data = dataset["train"].to_pandas()

# Define NLTK's stop word list as the gold standard
nltk_stop_words = set(stopwords.words('english'))

# Function to remove stop words using CoreNLP tokenization and NLTK's stop word list
def remove_stop_words_corenlp(text):
    with CoreNLPClient(annotators=['tokenize'], timeout=3000, memory='4G', port=9001,
                       start_server=StartServer.DONT_START) as client:
        ann = client.annotate(text)
        tokens = [token.word.lower() for sentence in ann.sentence for token in sentence.token]
        filtered_text = [word for word in tokens if word not in nltk_stop_words]
    return filtered_text  # Return tokens without stop words

# Function to extract actual stop words from the text using NLTK's list (gold standard)
def extract_stop_words(text):
    tokens = word_tokenize(text.lower())
    stop_words_in_text = [word for word in tokens if word in nltk_stop_words]
    return stop_words_in_text

# Define function to compare extracted stop words with gold standard
def evaluate_stop_word_removal(row):
    original_text = row['sms']
    removed_words = remove_stop_words_corenlp(original_text)
    gold_stop_words = extract_stop_words(original_text)

    # Convert lists to sets for comparison
    removed_set = set(word_tokenize(original_text.lower())) - set(removed_words)
    gold_set = set(gold_stop_words)

    # Calculate True Positives, False Positives, False Negatives
    tp = len(removed_set & gold_set)  # Correctly removed stop words
    fp = len(removed_set - gold_set)  # Incorrectly removed words (false positives)
    fn = len(gold_set - removed_set)  # Missed stop words (false negatives)
    tn = len(set(word_tokenize(original_text.lower())) - gold_set - removed_set)  # Correctly retained non-stop words

    # Calculate precision, recall, F1, and accuracy based on these counts
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return precision, recall, f1, accuracy

# Measure memory and time for the entire stop word removal process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply the evaluation to each row and compute metrics
metrics = data.apply(evaluate_stop_word_removal, axis=1)
precisions, recalls, f1_scores, accuracies = zip(*metrics)

# Calculate average metrics
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1_score = sum(f1_scores) / len(f1_scores)
avg_accuracy = sum(accuracies) / len(accuracies)

# Calculate time and memory usage
end_time = time.time()
memory_after = memory_usage()[0]
time_taken = end_time - start_time
memory_used = memory_after - memory_before

# Print evaluation results
print("Stop Word Removal Evaluation Metrics (using CoreNLP tokenization and NLTK Gold Standard):")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1_score:.4f}")
print(f"\nPerformance Metrics:")
print(f"Time taken for stop word removal: {time_taken:.4f} seconds")
print(f"Memory used for stop word removal: {memory_used:.4f} MB")
