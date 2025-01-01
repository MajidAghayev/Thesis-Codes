from datasets import load_dataset
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Load the SMS Spam dataset
dataset = load_dataset("ucirvine/sms_spam", trust_remote_code=True)
data = dataset["train"].to_pandas()

# Define stemming function using TextBlob
def stem_text(text):
    blob = TextBlob(text.lower())
    stemmed_text = ' '.join([word.stem() for word in blob.words])
    return stemmed_text

# Measure memory and time for stemming process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply stemming using TextBlob
data['stemmed_text'] = data['sms'].apply(stem_text)

# Calculate time and memory usage
end_time = time.time()
memory_after = memory_usage()[0]
time_taken = end_time - start_time
memory_used = memory_after - memory_before

# Define labels for spam classification
data['sentiment'] = data['label']  # 0 = ham, 1 = spam

# Split the data into train and test sets
X_train_orig, X_test_orig, y_train, y_test = train_test_split(data['sms'], data['sentiment'], test_size=0.2, random_state=42)
X_train_stem, X_test_stem, _, _ = train_test_split(data['stemmed_text'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the original and stemmed text data
vectorizer = CountVectorizer()
X_train_orig_vect = vectorizer.fit_transform(X_train_orig)
X_test_orig_vect = vectorizer.transform(X_test_orig)

X_train_stem_vect = vectorizer.fit_transform(X_train_stem)
X_test_stem_vect = vectorizer.transform(X_test_stem)

# Initialize the model
model_orig = MultinomialNB()
model_stem = MultinomialNB()

# Train and evaluate the model on original text
model_orig.fit(X_train_orig_vect, y_train)
y_pred_orig = model_orig.predict(X_test_orig_vect)

# Train and evaluate the model on stemmed text
model_stem.fit(X_train_stem_vect, y_train)
y_pred_stem = model_stem.predict(X_test_stem_vect)

# Calculate metrics for the original text model
accuracy_orig = accuracy_score(y_test, y_pred_orig)
precision_orig = precision_score(y_test, y_pred_orig)
recall_orig = recall_score(y_test, y_pred_orig)
f1_orig = f1_score(y_test, y_pred_orig)

# Calculate metrics for the stemmed text model
accuracy_stem = accuracy_score(y_test, y_pred_stem)
precision_stem = precision_score(y_test, y_pred_stem)
recall_stem = recall_score(y_test, y_pred_stem)
f1_stem = f1_score(y_test, y_pred_stem)

# Print metrics for comparison, formatted to four decimal places
print("Original Text Model Metrics:")
print(f"Accuracy: {accuracy_orig:.4f}")
print(f"Precision: {precision_orig:.4f}")
print(f"Recall: {recall_orig:.4f}")
print(f"F1 Score: {f1_orig:.4f}")

print("\nStemmed Text Model Metrics:")
print(f"Accuracy: {accuracy_stem:.4f}")
print(f"Precision: {precision_stem:.4f}")
print(f"Recall: {recall_stem:.4f}")
print(f"F1 Score: {f1_stem:.4f}")

# Print speed and memory usage results
print("\nStemming Performance Metrics:")
print(f"Time taken for stemming: {time_taken:.4f} seconds")
print(f"Memory used for stemming: {memory_used:.4f} MB")
