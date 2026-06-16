
import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
data = pd.read_csv("sample_reviews.csv")

# Text Cleaning
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

data["cleaned"] = data["review"].apply(clean_text)

# Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data["cleaned"]).toarray()
y = data["sentiment"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nAccuracy:", accuracy)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully.")
