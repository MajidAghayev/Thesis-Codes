from datasets import load_dataset
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download necessary NLTK resources
nltk.download('punkt')

# Load the SMS Spam dataset
dataset = load_dataset("ucirvine/sms_spam", trust_remote_code=True)
data = dataset["train"].to_pandas()


# Define text cleaning function
def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation, numbers, and special characters using regex
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Measure memory and time for the text cleaning process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply text cleaning
data['cleaned_text'] = data['sms'].apply(clean_text)

# Calculate time and memory usage
end_time = time.time()
memory_after = memory_usage()[0]
time_taken = end_time - start_time
memory_used = memory_after - memory_before

# Define labels for spam classification
data['sentiment'] = data['label']  # 0 = ham, 1 = spam

# Split the data into train and test sets
X_train_orig, X_test_orig, y_train, y_test = train_test_split(data['sms'], data['sentiment'], test_size=0.2,
                                                              random_state=42)
X_train_clean, X_test_clean, _, _ = train_test_split(data['cleaned_text'], data['sentiment'], test_size=0.2,
                                                     random_state=42)

# Vectorize the original and cleaned text data
vectorizer_orig = CountVectorizer()
X_train_orig_vect = vectorizer_orig.fit_transform(X_train_orig)
X_test_orig_vect = vectorizer_orig.transform(X_test_orig)

vectorizer_clean = CountVectorizer()
X_train_clean_vect = vectorizer_clean.fit_transform(X_train_clean)
X_test_clean_vect = vectorizer_clean.transform(X_test_clean)

# Initialize the model
model_orig = MultinomialNB()
model_clean = MultinomialNB()

# Train and evaluate the model on original text
model_orig.fit(X_train_orig_vect, y_train)
y_pred_orig = model_orig.predict(X_test_orig_vect)

# Train and evaluate the model on cleaned text
model_clean.fit(X_train_clean_vect, y_train)
y_pred_clean = model_clean.predict(X_test_clean_vect)

# Calculate metrics for the original text model
accuracy_orig = accuracy_score(y_test, y_pred_orig)
precision_orig = precision_score(y_test, y_pred_orig, zero_division=1)
recall_orig = recall_score(y_test, y_pred_orig, zero_division=1)
f1_orig = f1_score(y_test, y_pred_orig, zero_division=1)

# Calculate metrics for the cleaned text model
accuracy_clean = accuracy_score(y_test, y_pred_clean)
precision_clean = precision_score(y_test, y_pred_clean, zero_division=1)
recall_clean = recall_score(y_test, y_pred_clean, zero_division=1)
f1_clean = f1_score(y_test, y_pred_clean, zero_division=1)

# Print metrics for comparison, formatted to four decimal places
print("Original Text Model Metrics:")
print(f"Accuracy: {accuracy_orig:.4f}")
print(f"Precision: {precision_orig:.4f}")
print(f"Recall: {recall_orig:.4f}")
print(f"F1 Score: {f1_orig:.4f}")

print("\nCleaned Text Model Metrics:")
print(f"Accuracy: {accuracy_clean:.4f}")
print(f"Precision: {precision_clean:.4f}")
print(f"Recall: {recall_clean:.4f}")
print(f"F1 Score: {f1_clean:.4f}")

# Print speed and memory usage results
print("\nText Cleaning Performance Metrics:")
print(f"Time taken for text cleaning: {time_taken:.4f} seconds")
print(f"Memory used for text cleaning: {memory_used:.4f} MB")


