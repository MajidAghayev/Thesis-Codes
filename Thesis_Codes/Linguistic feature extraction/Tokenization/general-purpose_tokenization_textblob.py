from datasets import load_dataset
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Load the Amazon Subscription Boxes reviews subset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Subscription_Boxes", trust_remote_code=True)
data = dataset["full"].to_pandas()

# Define tokenization function using TextBlob
def tokenize_text_with_textblob(text):
    tokens = TextBlob(text.lower()).words  # Lowercase for consistency
    return ' '.join(tokens)

# Measure memory and time for the tokenization process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply TextBlob tokenization
data['tokenized_text'] = data['text'].apply(tokenize_text_with_textblob)

# Calculate time and memory usage
end_time = time.time()
memory_after = memory_usage()[0]
time_taken = end_time - start_time
memory_used = memory_after - memory_before

# Define labels for sentiment classification based on rating
# Ratings 4 and 5 are positive (1), and ratings 1, 2, and 3 are negative (0)
data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Split the data into train and test sets
X_train_orig, X_test_orig, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)
X_train_token, X_test_token, _, _ = train_test_split(data['tokenized_text'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the original and tokenized text data
vectorizer = CountVectorizer()
X_train_orig_vect = vectorizer.fit_transform(X_train_orig)
X_test_orig_vect = vectorizer.transform(X_test_orig)

X_train_token_vect = vectorizer.fit_transform(X_train_token)
X_test_token_vect = vectorizer.transform(X_test_token)

# Initialize the model
model_orig = MultinomialNB()
model_token = MultinomialNB()

# Train and evaluate the model on original text
model_orig.fit(X_train_orig_vect, y_train)
y_pred_orig = model_orig.predict(X_test_orig_vect)

# Train and evaluate the model on tokenized text
model_token.fit(X_train_token_vect, y_train)
y_pred_token = model_token.predict(X_test_token_vect)

# Calculate metrics for the original text model
accuracy_orig = accuracy_score(y_test, y_pred_orig)
precision_orig = precision_score(y_test, y_pred_orig)
recall_orig = recall_score(y_test, y_pred_orig)
f1_orig = f1_score(y_test, y_pred_orig)

# Calculate metrics for the tokenized text model
accuracy_token = accuracy_score(y_test, y_pred_token)
precision_token = precision_score(y_test, y_pred_token)
recall_token = recall_score(y_test, y_pred_token)
f1_token = f1_score(y_test, y_pred_token)

# Print metrics for comparison, formatted to four decimal places
print("Original Text Model Metrics:")
print(f"Accuracy: {accuracy_orig:.4f}")
print(f"Precision: {precision_orig:.4f}")
print(f"Recall: {recall_orig:.4f}")
print(f"F1 Score: {f1_orig:.4f}")

print("\nTokenized Text Model Metrics using TextBlob:")
print(f"Accuracy: {accuracy_token:.4f}")
print(f"Precision: {precision_token:.4f}")
print(f"Recall: {recall_token:.4f}")
print(f"F1 Score: {f1_token:.4f}")

# Print speed and memory usage results
print("\nTokenization Performance Metrics:")
print(f"Time taken for tokenization: {time_taken:.4f} seconds")
print(f"Memory used for tokenization: {memory_used:.4f} MB")
