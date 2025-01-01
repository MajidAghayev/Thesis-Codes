import nltk
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download necessary NLTK resources
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the correct Amazon Reviews 2023 dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Magazine_Subscriptions", trust_remote_code=True)
data = dataset["full"].to_pandas()


# Function to perform NER using NLTK
def nltk_ner(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    ne_tree = nltk.ne_chunk(pos_tags)  # Perform NER
    entities = []
    for chunk in ne_tree:
        if hasattr(chunk, 'label'):  # If it's an entity
            entity_name = " ".join(c[0] for c in chunk)  # Combine tokens
            entity_type = chunk.label()  # Get entity type
            entities.append((entity_name, entity_type))
    return entities

# Function to convert NLTK NER output to IOB tags
def nltk_to_iob(tokens, entities):
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
    accuracy = sum(1 for pred, silver in zip(pred_tags, silver_tags) if pred == silver) / len(silver_tags) if silver_tags else 0

    return accuracy, precision, recall, f1

# Initialize metrics storage
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Measure memory and time
start_time = time.time()
memory_before = memory_usage()[0]

# Evaluate self-consistency
for idx, row in data.iterrows():
    text = row["text"]

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Perform NER using NLTK (First Pass)
    pred_entities_first = nltk_ner(text)
    pred_iob_tags_first = nltk_to_iob(tokens, pred_entities_first)

    # Perform NER using NLTK (Second Pass)
    pred_entities_second = nltk_ner(text)
    pred_iob_tags_second = nltk_to_iob(tokens, pred_entities_second)

    # Calculate metrics for self-consistency
    accuracy, precision, recall, f1 = calculate_ner_metrics(pred_iob_tags_first, pred_iob_tags_second)

    # Append results
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Calculate average metrics
avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
avg_precision = sum(precisions) / len(precisions) if precisions else 0
avg_recall = sum(recalls) / len(recalls) if recalls else 0
avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

# Measure memory usage and time
memory_after = memory_usage()[0]
memory_used = memory_after - memory_before
time_taken = time.time() - start_time

# Print NER self-consistency metrics
print("NER Self-Consistency Metrics using NLTK on Amazon Reviews:")
print(f"Average Token-Level Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken: {time_taken:.4f} seconds")
print(f"Memory used: {memory_used:.4f} MB")
