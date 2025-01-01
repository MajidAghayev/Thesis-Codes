import stanza
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from memory_profiler import memory_usage

# Download and initialize Stanza with NER
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,ner', use_gpu=False)

# Load the CoNLL-2003 dataset
conll2003_dataset = load_dataset("conll2003", trust_remote_code=True)

# Mapping CoreNLP entity types to CoNLL-2003 tags
entity_mapping = {
    "PERSON": "PER",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "MISC": "MISC"
}

# Function to normalize CoreNLP NER output to IOB tags
def corenlp_to_iob(tokens, entities):
    bio_tags = ["O"] * len(tokens)  # Start with 'O' for all tokens

    for entity_text, entity_type in entities:
        if entity_type not in entity_mapping:
            continue
        entity_tokens = entity_text.split()
        for idx in range(len(tokens)):
            # Check if the entity matches starting from token idx
            if tokens[idx:idx + len(entity_tokens)] == entity_tokens:
                bio_tags[idx] = f"B-{entity_mapping[entity_type]}"
                for i in range(1, len(entity_tokens)):
                    if idx + i < len(tokens):
                        bio_tags[idx + i] = f"I-{entity_mapping[entity_type]}"
                break

    return bio_tags

# Function to calculate NER metrics
def calculate_ner_metrics(pred_tags, gold_tags):
    precision = precision_score(gold_tags, pred_tags, average='weighted', zero_division=0)
    recall = recall_score(gold_tags, pred_tags, average='weighted', zero_division=0)
    f1 = f1_score(gold_tags, pred_tags, average='weighted', zero_division=0)
    token_accuracy = accuracy_score(gold_tags, pred_tags)

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
    # Extract tokens and gold-standard tags
    tokens = document["tokens"]
    gold_tags_int = document["ner_tags"]
    gold_tags = [
        conll2003_dataset["train"].features["ner_tags"].feature.int2str(tag) for tag in gold_tags_int
    ]

    # Perform NER using CoreNLP
    text = " ".join(tokens)
    core_entities = [(ent.text, ent.type) for ent in nlp(text).entities]

    # Normalize CoreNLP entities to BIO tags
    pred_bio_tags = corenlp_to_iob(tokens, core_entities)

    # Skip mismatched samples
    if len(pred_bio_tags) != len(gold_tags):
        print(f"Skipping sample {idx}: Length mismatch (Predicted: {len(pred_bio_tags)}, Gold: {len(gold_tags)})")
        continue

    # Calculate metrics
    precision, recall, f1, token_accuracy = calculate_ner_metrics(pred_bio_tags, gold_tags)

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

# Print results
print("NER Metrics using CoreNLP with IOB tagging on CoNLL-2003 dataset:")
print(f"Average Token-Level Accuracy: {avg_token_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print("\nPerformance Metrics:")
print(f"Time taken for NER: {time_taken:.4f} seconds")
print(f"Memory used for NER: {memory_used:.4f} MB")
