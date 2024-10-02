from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define criteria and weights
criteria_keywords = {
    "NLP Experience": ["nlp", "bert", "spacy", "nltk", "sentiment analysis"],
    "Model Design": ["data", "cross-validation", "hyperparameters", "model", "logistic regression", "decision trees"],
    "Problem-Solving": ["challenging", "solved", "recommendation system", "unstructured"],
    "Communication Skills": ["communicate", "explained", "discussed", "clear", "concise"]
}

criteria_weights = {
    "NLP Experience": 1.5,
    "Model Design": 1.2,
    "Problem-Solving": 1.3,
    "Communication Skills": 1.0
}

# Sentiment Analysis Pipeline
nlp_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_candidate_responses():
    candidates = {}
    num_candidates = int(input("Enter the number of candidates: "))
    
    for _ in range(num_candidates):
        candidate_name = input("Enter candidate's name: ")
        responses = []
        
        # Collect responses for each criterion
        for criterion in criteria_keywords.keys():
            response = input(f"Enter {candidate_name}'s response for '{criterion}': ")
            responses.append(response)
        
        candidates[candidate_name] = responses
    
    return candidates

def evaluate_response(response, criteria_keywords, weight):
    try:
        lowercase_keywords = [word.lower() for word in criteria_keywords]
        vectorizer = TfidfVectorizer(vocabulary=lowercase_keywords)
        response_vector = vectorizer.fit_transform([response.lower()]).toarray()
        score = np.sum(response_vector)
        
        # Sentiment scoring
        sentiment_score = nlp_pipeline(response)[0]['score']
        
        # Apply weighting
        return score * sentiment_score * weight
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return 0

def rank_candidates(interview_responses, criteria_keywords, criteria_weights):
    candidate_scores = {}
    
    for candidate, responses in interview_responses.items():
        total_score = 0
        for response, criterion in zip(responses, criteria_keywords.keys()):
            weight = criteria_weights.get(criterion, 1)
            total_score += evaluate_response(response, criteria_keywords[criterion], weight)
        
        candidate_scores[candidate] = total_score
    
    ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_candidates

# Get
