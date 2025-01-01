import stanza
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download and initialize Stanza with POS tagging
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=False)

# Load the CoNLL-2003 dataset
conll2003_dataset = load_dataset("conll2003", trust_remote_code=True)

# Access the mapping for `pos_tags` from integers to strings
pos_tag_mapping = conll2003_dataset["train"].features["pos_tags"].feature.int2str

# Define mapping from Universal POS Tags (CoreNLP) to CoNLL-2003 POS tags
upos_to_conll_mapping = {
    "ADJ": "JJ",
    "ADP": "IN",
    "ADV": "RB",
    "AUX": "MD",
    "CCONJ": "CC",
    "DET": "DT",
    "INTJ": "UH",
    "NOUN": "NN",
    "NUM": "CD",
    "PART": "RP",
    "PRON": "PRP",
    "PROPN": "NNP",
    "PUNCT": ".",
    "SCONJ": "IN",
    "SYM": "SYM",
    "VERB": "VB",
    "X": "FW",
}

# Define Stanza POS tagging function with mapping
def corenlp_pos_tagging_and_mapping(text):
    doc = nlp(text)
    upos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]
    mapped_tags = [upos_to_conll_mapping.get(tag, "X") for tag in upos_tags]  # Default to "X" if no mapping exists
    return mapped_tags

# Function to calculate POS-level metrics by comparing with gold-standard POS tags
def calculate_pos_level_metrics(pred_tags, gold_tags):
    true_positives = sum(1 for pred, gold in zip(pred_tags, gold_tags) if pred == gold)
    false_positives = len(pred_tags) - true_positives
    false_negatives = len(gold_tags) - true_positives

    # POS-Level Accuracy calculation
    pos_level_accuracy = true_positives / len(gold_tags) if gold_tags else 0

    # Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return pos_level_accuracy, precision, recall, f1

# Initialize lists to store metrics for all sentences
pos_level_accuracies = []
precisions = []
recalls = []
f1_scores = []

# Measure memory and time for the POS tagging process
start_time = time.time()
memory_before = memory_usage()[0]

# Evaluate only the first 100 samples for efficiency
for idx, document in enumerate(conll2003_dataset["train"]):


    # Extract tokens and gold-standard POS tags
    tokens = document["tokens"]
    gold_tags_int = document["pos_tags"]

    # Convert integer POS tags to string representations
    gold_tags = [pos_tag_mapping(tag) for tag in gold_tags_int]

    # Concatenate tokens into a single string for CoreNLP tagging
    text = " ".join(tokens)

    # Tag the text using CoreNLP and map to CoNLL POS tags
    pred_tags = corenlp_pos_tagging_and_mapping(text)

    # Calculate POS-level accuracy, precision, recall, and F1 score
    pos_level_accuracy, precision, recall, f1 = calculate_pos_level_metrics(pred_tags, gold_tags)

    # Append metrics
    pos_level_accuracies.append(pos_level_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Calculate average metrics across all sentences
avg_pos_level_accuracy = sum(pos_level_accuracies) / len(pos_level_accuracies)
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1 = sum(f1_scores) / len(f1_scores)

# Measure memory usage after tagging
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before

# Calculate time taken for tagging
end_time = time.time()
time_taken = end_time - start_time

# Print POS tagging accuracy metrics
print("POS Tagging Metrics using CoreNLP on CoNLL-2003 dataset :")
print(f"Average POS-Level Accuracy: {avg_pos_level_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# Print performance metrics
print("\nPOS Tagging Performance Metrics:")
print(f"Time taken for tagging: {time_taken:.4f} seconds")
print(f"Memory used for tagging: {memory_used:.4f} MB")
