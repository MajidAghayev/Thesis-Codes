import stanza
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download and initialize Stanza with NER
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,ner', use_gpu=False)

# Load the Amazon Reviews 2023 dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Magazine_Subscriptions", trust_remote_code=True)
data = dataset["full"].to_pandas()


# CoreNLP NER function
def corenlp_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.type) for ent in doc.entities]

# Function to convert CoreNLP NER output to IOB tags
def corenlp_to_iob(tokens, entities):
    iob_tags = ["O"] * len(tokens)  # Start with 'O' for all tokens

    for entity_text, entity_type in entities:
        entity_tokens = entity_text.split()
        for idx in range(len(tokens)):
            # Check if the entity matches starting from token idx
            if tokens[idx:idx + len(entity_tokens)] == entity_tokens:
                iob_tags[idx] = f"B-{entity_type}"
                for i in range(1, len(entity_tokens)):
                    if idx + i < len(tokens):
                        iob_tags[idx + i] = f"I-{entity_type}"
                break

    return iob_tags

# Function to calculate NER metrics
def calculate_ner_metrics(pred_tags, silver_tags):
    precision = precision_score(silver_tags, pred_tags, average='weighted', zero_division=0)
    recall = recall_score(silver_tags, pred_tags, average='weighted', zero_division=0)
    f1 = f1_score(silver_tags, pred_tags, average='weighted', zero_division=0)
    accuracy = sum(1 for pred, silver in zip(pred_tags, silver_tags) if pred == silver) / len(silver_tags)

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
for idx, row in data.iterrows():
    text = row["text"]

    # Tokenize the text
    tokens = text.split()

    # Perform NER using CoreNLP
    pred_entities = corenlp_ner(text)

    # Convert to IOB format
    pred_iob_tags = corenlp_to_iob(tokens, pred_entities)

    # For silver standard, treat CoreNLP output as reference
    silver_iob_tags = pred_iob_tags  # Using the same output as reference

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_ner_metrics(pred_iob_tags, silver_iob_tags)

    # Append results
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

# Print NER evaluation metrics
print("NER Metrics using CoreNLP on Amazon Reviews 2023 dataset with  (IOB):")
print(f"Average Token-Level Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken for NER: {time_taken:.4f} seconds")
print(f"Memory used for NER: {memory_used:.4f} MB")
