# 📝 Homework: NLP Sentiment Analysis Practice

## 🎯 Assignment Overview

Create your own simple text classification application using the concepts learned in today's lesson. This homework will reinforce your understanding of NLP preprocessing, feature extraction, and model building.

**Time Estimate**: 2-3 hours  
**Difficulty Level**: Beginner  
**Due Date**: Next class session

## 📋 Task: Build a Movie Review Classifier

### **Objective**
Build a simple application that can classify movie reviews as either "Positive" or "Negative" using a different approach than the one shown in class.

### **What You Need to Do**

#### 1. **Data Collection** (30 minutes)
Create a dataset of 20 movie reviews:
- 10 positive reviews (you can find these from movie websites, or create realistic examples)
- 10 negative reviews
- Each review should be 1-3 sentences long
- Save them in a text file or directly in your Python code

**Example format:**
```python
reviews = [
    ("This movie was absolutely amazing! Great acting and storyline.", "positive"),
    ("Terrible movie, waste of time. Poor acting throughout.", "negative"),
    # ... add 18 more
]
```

#### 2. **Modify the Existing Code** (60 minutes)
Take the `sentiment_analyzer.py` file and make these changes:

**Required Changes:**
- [ ] Replace the sample data with your movie review dataset
- [ ] Add a new preprocessing step that removes numbers AND common movie-related stopwords (like "movie", "film", "watch")
- [ ] Change the output to display the actual review text along with the prediction
- [ ] Add a word count feature that shows how many words were in the original vs. cleaned text

**Code hints:**
```python
# Add custom stopwords
movie_stopwords = ["movie", "film", "watch", "see", "saw"]
self.stop_words.update(movie_stopwords)

# Add word count
original_word_count = len(text.split())
cleaned_word_count = len(cleaned_text.split())
```

#### 3. **Test Your Model** (30 minutes)
Test your classifier with these 5 new movie reviews:

```
1. "The plot was confusing and the ending made no sense at all."
2. "Brilliant cinematography and outstanding performances by all actors."
3. "This film exceeded all my expectations, highly recommended!"
4. "Boring dialogue and predictable storyline throughout the entire movie."
5. "A masterpiece of storytelling with incredible visual effects."
```

**Record the results** in a simple table:

| Review | Predicted Sentiment | Confidence | Your Assessment |
|--------|-------------------|------------|----------------|
| Review 1 | ? | ? | Correct/Incorrect |
| ... | ... | ... | ... |

#### 4. **Analysis and Reflection** (30 minutes)
Write a short paragraph (3-5 sentences) answering:
- How accurate was your model on the test reviews?
- What patterns did you notice in the predictions?
- What would you do differently to improve the model?

## 📁 Deliverables

Submit the following files:

1. **`movie_classifier.py`** - Your modified sentiment analyzer
2. **`test_results.txt`** - Your test results table and analysis paragraph
3. **`movie_reviews.txt`** - Your 20 movie reviews dataset (if saved separately)

## 🏆 Bonus Challenges (Optional)

**Easy Bonus (+5 points):**
- Add emoji output (😊 for positive, 😞 for negative)
- Count and display the most common words in positive vs negative reviews

**Medium Bonus (+10 points):**
- Implement a neutral category (for reviews that are neither clearly positive nor negative)
- Add 5 neutral movie reviews to your dataset

**Hard Bonus (+15 points):**
- Create a simple command-line menu system that lets users:
  1. Train the model
  2. Test a single review
  3. Test multiple reviews
  4. View model statistics
  5. Exit

## 📖 Learning Goals

By completing this homework, you will:
- Practice hands-on NLP implementation
- Understand the importance of domain-specific data
- Experience the iterative process of model improvement
- Gain confidence in modifying existing code
- Learn to evaluate model performance critically

## 🆘 Getting Help

**Stuck on something?**
1. Review the lesson materials (`LESSON.md`)
2. Check the main application code for examples
3. Ask classmates for general guidance (but don't share code)
4. Use online resources like Python documentation
5. Ask the instructor during office hours

**Common Issues:**
- **Import errors**: Make sure all packages are installed (`pip install -r requirements.txt`)
- **Low accuracy**: Normal with small datasets; focus on understanding the process
- **Code not running**: Check indentation and syntax carefully

## ✅ Submission Checklist

Before submitting, make sure:
- [ ] Your code runs without errors
- [ ] You've tested with all 5 provided reviews
- [ ] Your analysis paragraph is complete
- [ ] All files are named correctly
- [ ] You've added comments explaining your changes

## 📊 Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Data Collection | 25 | 20 realistic movie reviews with correct labels |
| Code Modifications | 40 | All required changes implemented correctly |
| Testing & Results | 20 | Complete test results table with all 5 reviews |
| Analysis | 15 | Thoughtful reflection on model performance |
| **Total** | **100** | |

---

**Good luck! 🍀 Remember, the goal is learning, not perfection. Focus on understanding the concepts rather than achieving high accuracy.**

*Questions? Check the class discussion forum or reach out during office hours.* 