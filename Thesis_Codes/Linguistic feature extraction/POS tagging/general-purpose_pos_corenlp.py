from datasets import load_dataset
import stanza
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download and initialize the CoreNLP model with Stanza for POS tagging
stanza.download('en')  # This only needs to be done once
nlp = stanza.Pipeline('en', processors='tokenize,pos')  # Initialize with tokenization and POS tagging

# Load the Amazon Subscription Boxes reviews subset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Subscription_Boxes", trust_remote_code=True)
data = dataset["full"].to_pandas()



# Define POS tagging function using CoreNLP
def pos_tag_text_with_corenlp(text):
    doc = nlp(text)
    pos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]
    return ' '.join(pos_tags)

# Measure memory and time for the POS tagging process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply CoreNLP POS tagging
data['pos_tags'] = data['text'].apply(pos_tag_text_with_corenlp)

# Calculate time and memory usage
end_time = time.time()
memory_after = memory_usage()[0]
time_taken = end_time - start_time
memory_used = memory_after - memory_before

# Define labels for sentiment classification based on rating
# Ratings 4 and 5 are positive (1), and ratings 1, 2, and 3 are negative (0)
data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Split the data into train and test sets
X_train_pos, X_test_pos, y_train, y_test = train_test_split(data['pos_tags'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the POS-tagged text data
vectorizer = CountVectorizer()
X_train_pos_vect = vectorizer.fit_transform(X_train_pos)
X_test_pos_vect = vectorizer.transform(X_test_pos)

# Initialize the model
model_pos = MultinomialNB()

# Train and evaluate the model on POS-tagged text
model_pos.fit(X_train_pos_vect, y_train)
y_pred_pos = model_pos.predict(X_test_pos_vect)

# Calculate metrics for the POS-tagged text model
accuracy_pos = accuracy_score(y_test, y_pred_pos)
precision_pos = precision_score(y_test, y_pred_pos)
recall_pos = recall_score(y_test, y_pred_pos)
f1_pos = f1_score(y_test, y_pred_pos)

# Print metrics for POS-tagged text model
print("POS-Tagged Text Model Metrics:")
print(f"Accuracy: {accuracy_pos:.4f}")
print(f"Precision: {precision_pos:.4f}")
print(f"Recall: {recall_pos:.4f}")
print(f"F1 Score: {f1_pos:.4f}")

# Print speed and memory usage results
print("\nPOS Tagging Performance Metrics:")
print(f"Time taken for POS tagging: {time_taken:.4f} seconds")
print(f"Memory used for POS tagging: {memory_used:.4f} MB")
