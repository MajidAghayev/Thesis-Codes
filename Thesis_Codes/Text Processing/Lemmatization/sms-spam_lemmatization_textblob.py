from datasets import load_dataset
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Load the SMS Spam dataset
dataset = load_dataset("ucirvine/sms_spam")
data = dataset["train"].to_pandas()

# Define lemmatization function using TextBlob
def lemmatize_text_with_textblob(text):
    tokens = text.lower().split()
    lemmatized_tokens = [Word(token).lemmatize() for token in tokens]
    return ' '.join(lemmatized_tokens)

# Measure memory and time for lemmatization process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply TextBlob lemmatization
data['lemmatized_text'] = data['sms'].apply(lemmatize_text_with_textblob)

# Calculate time and memory usage
end_time = time.time()
memory_after = memory_usage()[0]
time_taken = end_time - start_time
memory_used = memory_after - memory_before

# Define labels for spam classification
data['sentiment'] = data['label']  # 0 = ham, 1 = spam

# Split the data into train and test sets
X_train_orig, X_test_orig, y_train, y_test = train_test_split(data['sms'], data['sentiment'], test_size=0.2, random_state=42)
X_train_lemma, X_test_lemma, _, _ = train_test_split(data['lemmatized_text'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the original and lemmatized text data
vectorizer = CountVectorizer()
X_train_orig_vect = vectorizer.fit_transform(X_train_orig)
X_test_orig_vect = vectorizer.transform(X_test_orig)

X_train_lemma_vect = vectorizer.fit_transform(X_train_lemma)
X_test_lemma_vect = vectorizer.transform(X_test_lemma)

# Initialize the model
model_orig = MultinomialNB()
model_lemma = MultinomialNB()

# Train and evaluate the model on original text
model_orig.fit(X_train_orig_vect, y_train)
y_pred_orig = model_orig.predict(X_test_orig_vect)

# Train and evaluate the model on lemmatized text
model_lemma.fit(X_train_lemma_vect, y_train)
y_pred_lemma = model_lemma.predict(X_test_lemma_vect)

# Calculate metrics for the original text model
accuracy_orig = accuracy_score(y_test, y_pred_orig)
precision_orig = precision_score(y_test, y_pred_orig)
recall_orig = recall_score(y_test, y_pred_orig)
f1_orig = f1_score(y_test, y_pred_orig)

# Calculate metrics for the lemmatized text model
accuracy_lemma = accuracy_score(y_test, y_pred_lemma)
precision_lemma = precision_score(y_test, y_pred_lemma)
recall_lemma = recall_score(y_test, y_pred_lemma)
f1_lemma = f1_score(y_test, y_pred_lemma)

# Print metrics for comparison, formatted to four decimal places
print("Original Text Model Metrics:")
print(f"Accuracy: {accuracy_orig:.4f}")
print(f"Precision: {precision_orig:.4f}")
print(f"Recall: {recall_orig:.4f}")
print(f"F1 Score: {f1_orig:.4f}")

print("\nLemmatized Text Model Metrics using TextBlob:")
print(f"Accuracy: {accuracy_lemma:.4f}")
print(f"Precision: {precision_lemma:.4f}")
print(f"Recall: {recall_lemma:.4f}")
print(f"F1 Score: {f1_lemma:.4f}")

# Print speed and memory usage results
print("\nLemmatization Performance Metrics:")
print(f"Time taken for lemmatization: {time_taken:.4f} seconds")
print(f"Memory used for lemmatization: {memory_used:.4f} MB")
