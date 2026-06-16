
from flask import Flask, request, jsonify
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

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

@app.route("/")
def home():
    return "Sentiment Analysis API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned]).toarray()

    prediction = model.predict(vector)[0]

    return jsonify({
        "input": text,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
