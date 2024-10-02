from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Mock interview responses
interview_responses = {
    "Candidate A": [
        "I have extensive experience with NLP. I’ve worked with libraries like NLTK, spaCy, and built models using BERT for sentiment analysis tasks.",
        "I start by understanding the problem, collecting relevant data, and then choosing a model that fits. I use techniques like cross-validation to tune hyperparameters.",
        "One of the most challenging AI problems I solved was implementing a recommendation system for a retail client, where data was highly unstructured."
    ],
    "Candidate B": [
        "I have some experience in NLP, mostly with text classification using simple models like Naive Bayes.",
        "I usually go with a straightforward approach, selecting commonly used models like logistic regression or decision trees.",
        "I haven't faced too many challenging AI problems yet, as I’m still exploring the field."
    ]
}

# Predefined criteria and sample answers (converted to lowercase for matching)
criteria_keywords = {
    "NLP Experience": ["nlp", "bert", "spacy", "nltk", "sentiment analysis"],
    "Model Design": ["data", "cross-validation", "hyperparameters", "model", "logistic regression", "decision trees"],
    "Problem-Solving": ["challenging", "solved", "recommendation system", "unstructured"],
    "Communication Skills": ["communicate", "explained", "discussed", "clear", "concise"]
}

# Define weights for each criterion
criteria_weights = {
    "NLP Experience": 1.5,
    "Model Design": 1.2,
    "Problem-Solving": 1.3,
    "Communication Skills": 1.0
}

# Sentiment Analysis Pipeline (specifying the model to avoid warnings)
nlp_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def evaluate_response(response, criteria_keywords, weight):
    try:
        # Convert criteria keywords to lowercase for matching
        lowercase_keywords = [word.lower() for word in criteria_keywords]
        vectorizer = TfidfVectorizer(vocabulary=lowercase_keywords)
        response_vector = vectorizer.fit_transform([response.lower()]).toarray()  # Lowercase the response for matching
        score = np.sum(response_vector)
        
        # Sentiment scoring
        sentiment_score = nlp_pipeline(response)[0]['score']
        
        # Apply weighting
        return score * sentiment_score * weight
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return 0  # Return a neutral score if an error occurs

def rank_candidates(interview_responses, criteria_keywords, criteria_weights):
    candidate_scores = {}
    
    for candidate, responses in interview_responses.items():
        total_score = 0
        for response, criterion in zip(responses, criteria_keywords.keys()):
            weight = criteria_weights.get(criterion, 1)  # Default weight is 1 if not specified
            total_score += evaluate_response(response, criteria_keywords[criterion], weight)
        
        candidate_scores[candidate] = total_score
    
    # Rank candidates by total score
    ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_candidates

# Evaluate and rank candidates
ranked_candidates = rank_candidates(interview_responses, criteria_keywords, criteria_weights)

# Print results
for candidate, score in ranked_candidates:
    print(f"{candidate}: {score:.2f}")
