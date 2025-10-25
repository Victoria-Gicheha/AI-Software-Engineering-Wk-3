# NLP with spaCy: Named Entity Recognition + Sentiment Analysis 
# Goal: Extract product/brand entities and analyze sentiment

import spacy


# 1. Load spaCy's English model 
nlp = spacy.load("en_core_web_sm")


# 2. Sample Amazon product reviews
reviews = [
    "I absolutely love my new Apple iPhone! The camera quality is amazing.",
    "The Samsung Galaxy screen cracked within a week. Totally disappointed.",
    "This Logitech mouse is super smooth and comfortable to use.",
    "I hate the battery life on my Dell laptop. It barely lasts 2 hours!",
    "The Sony headphones have great sound but the build quality feels cheap."
]




# 3. Simple rule-based sentiment analyzer 
positive_words = ["love", "great", "amazing", "excellent", "super", "smooth", "happy"]
negative_words = ["hate", "disappointed", "bad", "poor", "cheap", "terrible", "cracked"]

def analyze_sentiment(text):
    text_lower = text.lower()
    pos = sum(word in text_lower for word in positive_words)
    neg = sum(word in text_lower for word in negative_words)
    if pos > neg:
        return "Positive 😊"
    elif neg > pos:
        return "Negative 😞"
    else:
        return "Neutral 😐"



# 4. Process each review 
for review in reviews:
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = analyze_sentiment(review)
    
    print("📝 Review:", review)
    print("🔹 Named Entities:", entities if entities else "None found")
    print("💬 Sentiment:", sentiment)
    print("-" * 60)
