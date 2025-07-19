"""
Build a simple application that can classify movie reviews as either "Positive" or "Negative" using a different approach than the one shown in class.

What You Need to Do
1. Data Collection
Create a dataset of 20 movie reviews:
1.10 positive reviews (you can find these from movie websites, or create realistic examples)
2.10 negative reviews
3.Each review should be 1-3 sentences long
4.Save them in a text file or directly in your Python code

Example format:
reviews = [
    ("This movie was absolutely amazing! Great acting and storyline.", "positive"),
    ("Terrible movie, waste of time. Poor acting throughout.", "negative"),
    # ... add 18 more
]

2. Modify the Existing Code
Take the sentiment_analyzer.py file and make these changes:
Required Changes:
1.Replace the sample data with your movie review dataset
2.Add a new preprocessing step that removes numbers AND common movie-related stopwords (like "movie", "film", "watch")
3.Change the output to display the actual review text along with the prediction
4.Add a word count feature that shows how many words were in the original vs. cleaned text

Code hints:
# Add custom stopwords
movie_stopwords = ["movie", "film", "watch", "see", "saw"]
self.stop_words.update(movie_stopwords)

# Add word count
original_word_count = len(text.split())
cleaned_word_count = len(cleaned_text.split())

3. Test Your Model
Test your classifier with these 5 new movie reviews:
1. "The plot was confusing and the ending made no sense at all."
2. "Brilliant cinematography and outstanding performances by all actors."
3. "This film exceeded all my expectations, highly recommended!"
4. "Boring dialogue and predictable storyline throughout the entire movie."
5. "A masterpiece of storytelling with incredible visual effects."

4. Analysis and Reflection
Write a short paragraph (3-5 sentences) answering:
1.How accurate was your model on the test reviews?
2.What patterns did you notice in the predictions?
3.What would you do differently to improve the model?

Deliverables
Submit the following files:
1.movie_classifier.py - Your modified sentiment analyzer
2.test_results.txt - Your test results table and analysis paragraph
3.movie_reviews.txt - Your 20 movie reviews dataset (if saved separately)

"""