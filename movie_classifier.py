import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from collections import Counter

# Ensure NLTK resources are available
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}") if resource == "punkt" else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

movie_stopwords = {"movie", "film", "watch", "see", "saw"}

class MovieReviewClassifier:
    def __init__(self):
        self.stop_words = set(stopwords.words("english")) | movie_stopwords
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ("nb", MultinomialNB()),
        ])

    def clean_text(self, text: str) -> tuple[str, int, int]:
        original_word_count = len(text.split())
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        cleaned_text = " ".join(tokens)
        cleaned_word_count = len(cleaned_text.split())
        return cleaned_text, original_word_count, cleaned_word_count

    def prepare_data(self, reviews: list[tuple[str, str]]) -> pd.DataFrame:
        df = pd.DataFrame(reviews, columns=["text", "label"])
        df["cleaned_text"], df["original_wc"], df["cleaned_wc"] = zip(*df["text"].apply(self.clean_text))
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df["label"] = df["label"].map(label_map)
        return df

    def train(self, df: pd.DataFrame) -> float:
        X = df["cleaned_text"]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)

    def predict(self, text: str) -> dict:
        cleaned_text, orig_wc, clean_wc = self.clean_text(text)
        prediction = self.model.predict([cleaned_text])[0]
        if prediction == 2:
            sentiment = "Positive"
        elif prediction == 1:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        prob = self.model.predict_proba([cleaned_text])[0].max()
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "original_wc": orig_wc,
            "cleaned_wc": clean_wc,
            "sentiment": sentiment,
            "confidence": prob,
        }

def format_results(classifier: "MovieReviewClassifier", reviews: list[str]) -> str:
    """Return formatted prediction results for the given reviews."""
    lines = []
    for text in reviews:
        result = classifier.predict(text)
        if result["sentiment"] == "Positive":
            emoji = "😊"
        elif result["sentiment"] == "Negative":
            emoji = "😞"
        else:
            emoji = "😐"
        lines.append(
            f"Review: {result['text']}\n"
            f"Prediction: {result['sentiment']} {emoji} (Confidence: {result['confidence']:.2f})\n"
            f"Word Count: original={result['original_wc']} cleaned={result['cleaned_wc']}\n"
        )
    return "\n".join(lines)


def save_results(text: str, filename: str = "test_result.txt"):
    """Save the given text to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def display_common_words(classifier: "MovieReviewClassifier", reviews: list[tuple[str, str]], top_n: int = 3) -> None:
    """Print the most common words in positive, neutral, and negative reviews."""
    pos_words: list[str] = []
    neg_words: list[str] = []
    neu_words: list[str] = []
    for text, label in reviews:
        cleaned, _, _ = classifier.clean_text(text)
        if label == "positive":
            pos_words.extend(cleaned.split())
        elif label == "neutral":
            neu_words.extend(cleaned.split())
        else:
            neg_words.extend(cleaned.split())

    pos_common = ", ".join(w for w, _ in Counter(pos_words).most_common(top_n))
    neg_common = ", ".join(w for w, _ in Counter(neg_words).most_common(top_n))
    neu_common = ", ".join(w for w, _ in Counter(neu_words).most_common(top_n))

    print(f"\nMost common words in positive reviews: {pos_common}")
    print(f"Most common words in neutral reviews: {neu_common}")
    print(f"Most common words in negative reviews: {neg_common}")


def main():
    reviews = [
        ("An amazing movie with stunning visuals and a heartfelt story.", "positive"),
        ("Absolutely loved the performances and the storyline was gripping.", "positive"),
        ("The film was a delightful surprise and kept me engaged throughout.", "positive"),
        ("Fantastic direction and brilliant acting made this film great.", "positive"),
        ("A wonderful experience that I would happily watch again.", "positive"),
        ("Brilliant cinematography and outstanding music score.", "positive"),
        ("A masterpiece that showcases the best of filmmaking.", "positive"),
        ("Incredible storytelling with characters you truly care about.", "positive"),
        ("An uplifting film that exceeded all of my expectations.", "positive"),
        ("Heartwarming and entertaining from start to finish.", "positive"),
        ("The movie had some good moments but overall it was just average.", "neutral"),
        ("Parts of the film were interesting though others felt dull.", "neutral"),
        ("A mix of compelling scenes and lackluster storytelling.", "neutral"),
        ("Some performances stood out, but the plot didn't leave a strong impression.", "neutral"),
        ("Neither great nor terrible, it was simply okay.", "neutral"),
        ("A fair attempt with some memorable scenes but ultimately unremarkable.", "neutral"),
        ("Nothing groundbreaking yet not a waste of time either.", "neutral"),
        ("Average pacing and a storyline that neither thrilled nor bored.", "neutral"),
        ("Some parts were enjoyable though others dragged on.", "neutral"),
        ("It was okay overall but lacked any standout qualities.", "neutral"),
        ("The plot was confusing and the pacing was painfully slow.", "negative"),
        ("Terrible acting and a script that made no sense at all.", "negative"),
        ("A disappointing film with a predictable storyline.", "negative"),
        ("Poorly executed with characters that felt flat and uninteresting.", "negative"),
        ("A forgettable movie that failed to hold my attention.", "negative"),
        ("Boring dialogue and a complete waste of time.", "negative"),
        ("Weak plot development and awful special effects.", "negative"),
        ("The worst film I have seen in years.", "negative"),
        ("Unconvincing performances and terrible editing.", "negative"),
        ("Predictable from the opening scene to the final shot.", "negative"),
    ]

    classifier = MovieReviewClassifier()
    df = classifier.prepare_data(reviews)
    accuracy = classifier.train(df)
    print(f"Model trained with accuracy: {accuracy:.2f}")

    display_common_words(classifier, reviews)

    test_reviews = [
        "The plot was confusing and the ending made no sense at all.",
        "Brilliant cinematography and outstanding performances by all actors.",
        "This film exceeded all my expectations, highly recommended!",
        "Boring dialogue and predictable storyline throughout the entire movie.",
        "A masterpiece of storytelling with incredible visual effects.",
        "The movie was acceptable, neither exciting nor disappointing.",
        "Some scenes were interesting, but the film as a whole felt ordinary.",
        "It was a decent watch, though not particularly memorable moments."
    ]

    print("\nTest Results:")
    results_text = format_results(classifier, test_reviews)
    print(results_text)
    save_results(results_text)


if __name__ == "__main__":
    main()