
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import nltk
import re

nltk.download('punkt')

text1 = '''
Machine learning improves healthcare systems.
'''

text2 = '''
Machine learning helps healthcare systems improve.
'''
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

clean1 = clean_text(text1)
clean2 = clean_text(text2)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectors = vectorizer.fit_transform([clean1, clean2])

cosine_score = cosine_similarity(vectors)[0][1]
fuzzy_score = fuzz.ratio(clean1, clean2)

plagiarism_percent = (cosine_score * 100 + fuzzy_score) / 2

print("\nPlagiarism Detection Report")
print("--------------------------------")
print("Cosine Similarity:", round(cosine_score * 100, 2), "%")
print("Fuzzy Matching:", round(fuzzy_score, 2), "%")
print("Estimated Plagiarism:", round(plagiarism_percent, 2), "%")

if plagiarism_percent > 60:
    print("\nPotential Plagiarism Detected")
else:
    print("\nTexts are Different")
