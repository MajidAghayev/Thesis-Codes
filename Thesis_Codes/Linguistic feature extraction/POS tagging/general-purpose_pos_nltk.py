from datasets import load_dataset
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from memory_profiler import memory_usage
import time

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Subscription_Boxes", trust_remote_code=True)
data = dataset["full"].to_pandas()  

# Define POS tagging function using NLTK
def pos_tag_nltk(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([tag for _, tag in nltk.pos_tag(tokens)])

# Measure memory and time for the POS tagging process
start_time = time.time()
memory_before = memory_usage()[0]

# Apply NLTK POS tagging
data['pos_tags'] = data['text'].apply(pos_tag_nltk)

memory_after = memory_usage()[0]
time_taken = time.time() - start_time
memory_used = memory_after - memory_before

# Define labels for sentiment classification based on rating
# Ratings 4 and 5 are positive (1), and ratings 1, 2, and 3 are negative (0)
data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['pos_tags'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the POS-tagged text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics for the NLTK POS-tagged text model
print("NLTK POS Tagging Metrics:")
print(f"Time taken: {time_taken:.4f} seconds")
print(f"Memory used: {memory_used:.4f} MB")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
