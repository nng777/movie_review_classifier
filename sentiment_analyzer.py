import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        self.model = None
        self.vectorizer = None
        
    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join back to string
        return ' '.join(tokens)
    
    def create_sample_data(self):
        """
        Create sample training data for demonstration
        """
        # Sample data with labels (1 = positive, 0 = negative)
        data = [
            ("I love this movie, it's absolutely fantastic!", 1),
            ("This is the best day ever!", 1),
            ("Amazing product, highly recommend!", 1),
            ("Great service and friendly staff", 1),
            ("Wonderful experience, will come back", 1),
            ("This is terrible, worst experience ever", 0),
            ("I hate this product, waste of money", 0),
            ("Awful service, very disappointed", 0),
            ("This movie is boring and pointless", 0),
            ("Poor quality, would not recommend", 0),
            ("The weather is okay today", 1),  # Neutral-positive
            ("This is an average product", 0),   # Neutral-negative
            ("The book was interesting and well-written", 1),
            ("Staff was helpful but the food was cold", 0),
            ("Good value for money, satisfied with purchase", 1)
        ]
        
        df = pd.DataFrame(data, columns=['text', 'sentiment'])
        return df
    
    def train_model(self, df):
        """
        Train the sentiment analysis model
        """
        # Clean the text data
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Create features and labels
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given text
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Predict using trained model
        prediction = self.model.predict([cleaned_text])[0]
        probability = self.model.predict_proba([cleaned_text])[0]
        
        # Also get VADER sentiment scores for comparison
        vader_scores = self.sia.polarity_scores(text)
        
        # Interpret results
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability)
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_sentiment': sentiment_label,
            'confidence': confidence,
            'vader_compound': vader_scores['compound'],
            'vader_interpretation': self._interpret_vader_score(vader_scores['compound'])
        }
    
    def _interpret_vader_score(self, compound_score):
        """
        Interpret VADER compound score
        """
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def analyze_batch(self, texts):
        """
        Analyze multiple texts at once
        """
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results
    
    def save_model(self, filename='sentiment_model.pkl'):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='sentiment_model.pkl'):
        """
        Load a pre-trained model
        """
        if os.path.exists(filename):
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found!")

def main():
    """
    Main function to demonstrate the sentiment analyzer
    """
    print("🤖 Simple Sentiment Analyzer - NLP in Action!")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimpleSentimentAnalyzer()
    
    # Create and use sample data
    print("📊 Creating sample training data...")
    df = analyzer.create_sample_data()
    print(f"Created dataset with {len(df)} samples")
    
    # Train model
    print("\n🚀 Training sentiment analysis model...")
    accuracy = analyzer.train_model(df)
    
    # Demo predictions
    print("\n🎯 Testing predictions:")
    test_texts = [
        "I absolutely love this new product!",
        "This is the worst service I've ever experienced.",
        "The movie was okay, nothing special.",
        "Fantastic work, keep it up!",
        "I'm feeling great today!"
    ]
    
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"\nText: '{result['text']}'")
        print(f"Prediction: {result['predicted_sentiment']} (Confidence: {result['confidence']:.2f})")
        print(f"VADER Score: {result['vader_compound']:.2f} ({result['vader_interpretation']})")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("🔍 Interactive Mode - Try your own text!")
    print("(Type 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter text to analyze: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            try:
                result = analyzer.predict_sentiment(user_input)
                print(f"Sentiment: {result['predicted_sentiment']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"VADER: {result['vader_interpretation']} ({result['vader_compound']:.2f})")
            except Exception as e:
                print(f"Error: {e}")
    
    print("\n👋 Thanks for using the Sentiment Analyzer!")

if __name__ == "__main__":
    main() 