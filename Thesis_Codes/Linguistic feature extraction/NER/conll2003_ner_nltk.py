import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time
from stanza.server import CoreNLPClient
import nltk

# Load the correct Amazon Reviews 2023 dataset
conll2003_dataset = load_dataset("conll2003", trust_remote_code=True)

# Define mapping from integer tags to string labels
tag_mapping = conll2003_dataset["train"].features["ner_tags"].feature.int2str

# Function to tokenize, POS tag, and perform NER using NLTK with IOB tagging
def nltk_ner_iob(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    ne_tree = nltk.ne_chunk(pos_tags)  # Perform NER

    iob_tags = []
    for subtree in ne_tree:
        if isinstance(subtree, nltk.Tree):
            entity_label = subtree.label()
            for idx, (word, pos) in enumerate(subtree):
                tag = f"B-{entity_label}" if idx == 0 else f"I-{entity_label}"
                iob_tags.append((word, tag))
        else:
            word, pos = subtree
            iob_tags.append((word, "O"))

    return iob_tags

# Function to convert CoNLL-2003 tags to IOB format
def convert_conll_to_iob(tokens, tags, tag_mapping):
    iob_tags = []
    for token, tag in zip(tokens, tags):
        tag_str = tag_mapping(tag)  # Convert integer to string label
        if tag_str.startswith("B") or tag_str.startswith("I"):
            iob_tags.append((token, tag_str))
        else:
            iob_tags.append((token, "O"))
    return iob_tags

# Function to calculate NER metrics
def calculate_ner_metrics(pred_tags, gold_tags):
    pred_flat = [tag for _, tag in pred_tags]
    gold_flat = [tag for _, tag in gold_tags]

    precision = precision_score(gold_flat, pred_flat, average='weighted', zero_division=0)
    recall = recall_score(gold_flat, pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(gold_flat, pred_flat, average='weighted', zero_division=0)
    token_accuracy = accuracy_score(gold_flat, pred_flat)

    return precision, recall, f1, token_accuracy

# Initialize metrics storage
precisions = []
recalls = []
f1_scores = []
token_accuracies = []

# Measure memory and time
start_time = time.time()
memory_before = memory_usage()[0]


for idx, document in enumerate(conll2003_dataset["train"]):
    # Get tokens and gold-standard NER tags
    tokens = document["tokens"]
    gold_tags_int = document["ner_tags"]
    gold_tags = convert_conll_to_iob(tokens, gold_tags_int, tag_mapping)

    # Perform NER using NLTK
    text = " ".join(tokens)
    pred_ner = nltk_ner_iob(text)

    # Skip mismatched samples
    if len(pred_ner) != len(gold_tags):
        print(f"Skipping sample {idx}: Length mismatch (Predicted: {len(pred_ner)}, Gold: {len(gold_tags)})")
        continue

    # Calculate metrics
    precision, recall, f1, token_accuracy = calculate_ner_metrics(pred_ner, gold_tags)

    # Append results
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    token_accuracies.append(token_accuracy)

# Calculate average metrics
avg_precision = sum(precisions) / len(precisions) if precisions else 0
avg_recall = sum(recalls) / len(recalls) if recalls else 0
avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
avg_token_accuracy = sum(token_accuracies) / len(token_accuracies) if token_accuracies else 0

# Measure memory usage and time
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before
time_taken = time.time() - start_time

# Print NER evaluation metrics
print("NER Metrics using NLTK with IOB tagging on CoNLL-2003 dataset:")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average Token-Level Accuracy: {avg_token_accuracy:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken for NER: {time_taken:.4f} seconds")
print(f"Memory used for NER: {memory_used:.4f} MB")
