
# AI Sentiment Analysis Model

## Overview
This project is a complete AI-based Sentiment Analysis system that classifies text into:
- Positive
- Negative
- Neutral

The project uses:
- Python
- Scikit-learn
- NLTK
- Pandas
- TF-IDF Vectorization
- Logistic Regression

## Features
- Text preprocessing
- Stopword removal
- Lemmatization
- TF-IDF feature extraction
- ML model training
- Accuracy & F1-score evaluation
- Prediction on custom text
- Simple deployment-ready Flask API

## Project Structure
- `train_model.py` → Train the sentiment model
- `app.py` → Flask deployment app
- `sample_reviews.csv` → Example dataset
- `requirements.txt` → Required libraries
- `README.md` → Documentation

## How to Run

### Install Libraries
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train_model.py
```

### Run Flask App
```bash
python app.py
```

## Example Prediction
Input:
```text
The food was amazing and service was excellent.
```

Output:
```text
Positive
```
