# Week 1: NLP Fundamentals & Lexicon-Based Sentiment Analysis

In Week 1, you'll build your first sentiment analysis system using **lexicon-based approaches**. This is the foundation for the entire project.

**What you'll build:**
- Text preprocessing pipeline (tokenization, stemming, lemmatization)
- Sentiment analyzer using VADER + TextBlob + custom lexicon
- Evaluation on real Financial PhraseBank dataset
- Confusion matrix and classification metrics

---

## Three Tasks This Week

### Task 1: Text Preprocessing Pipeline 

**What:** Build a `TextPreprocessor` class that cleans financial text

**Learn:** Tokenization, stop words, stemming, lemmatization

**Output:** Working class that turns:
- Input: `"Apple Inc. shares fell 2.5% amid concerns about iPhone 15 demand."`
- Output: `['apple', 'inc', 'shares', 'fell', '2.5', 'amid', 'concern', 'iphone', '15', 'demand']`

---

### Task 2: Sentiment Scoring 

**What:** Implement three sentiment scoring methods

**Learn:** How VADER, TextBlob, and custom lexicon approaches work

**Output:** Single sentence scored by all three methods:
- VADER compound: -1 to +1
- TextBlob polarity: 0 to 1
- Custom score: -1 to +1
- Ensemble average: consensus score

**Example:**
```
"Excellent earnings report beat expectations"
- VADER: 0.82 (positive)
- TextBlob: 0.75 (positive)
- Custom: 0.80 (positive)
- Ensemble: 0.79 (positive) ← Most reliable
```

---

### Task 3: Evaluation on Real Data 

**What:** Load 4,844 labeled sentences, evaluate your sentiment system

**Learn:** How to calculate precision, recall, F1-score, confusion matrix

**Output:** Performance metrics:
- Accuracy
- Per-class precision/recall/F1
- Confusion matrix visualization

---

## Resources & Learning Materials

### Useful Articles 

1. **Text Preprocessing for NLP** 
   - https://medium.com/@datamonkeysk/text-preprocessing-for-nlp-6d100f6546a6
   - Learn: tokenization, stemming, lemmatization trade-offs

2. **VADER Sentiment Analysis** 
   - https://realpython.com/python-nltk-sentiment-analysis/
   - Learn: How VADER compound scores work, when to use it

3. **Classification Metrics Explained** 
   - https://towardsdatascience.com/precision-recall-and-f1-score-4d27eb3c7b1
   - Learn: TP/FP/TN/FN, precision, recall, F1-score

### Useful Videos

1. **NLP Preprocessing** 
   - https://www.youtube.com/watch?v=kQZ8UGSJohg
   - Visual walkthrough of preprocessing steps

2. **VADER Sentiment Analysis** 
   - YouTube: "VADER sentiment analysis tutorial"
   - How compound scores are calculated

3. **Confusion Matrix & Metrics** 
   - https://www.youtube.com/watch?v=Kdsp6soqA7o
   - StatQuest visual explanation (excellent!)

4. **Classification Metrics** 
   - YouTube: "precision recall F1-score explained"

### Reference Docs

- **NLTK:** https://www.nltk.org/
- **TextBlob:** https://textblob.readthedocs.io/
- **scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Dataset

### [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

---

## Three Jupyter Notebooks to Submit

### Notebook 1: task_1_text_preprocessing.ipynb

```python
# [CODE] Implement TextPreprocessor class
# [TESTS] Test on sample sentences

class TextPreprocessor:
    def expand_contractions(text)
    def remove_special_chars(text)
    def tokenize(text)
    def remove_stopwords(tokens)
    def stem(tokens)
    def lemmatize(tokens)
    def preprocess(text, use_lemmatization=False)

# Expected output:
# Input: "Apple Inc. shares fell 2.5% amid market concerns"
# Output: ['apple', 'inc', 'shares', 'fell', '2.5', 'amid', 'market', 'concern']
```

**Acceptance Criteria:**
- Tokenization works
- Stop words removed (keeping financial terms)
- Stemming/lemmatization applied
- Handles special chars and contractions

---

### Notebook 2: task_2_lexicon_sentiment.ipynb

```python
#  [CODE] Implement three sentiment methods
#    - VADER sentiment analyzer
#    - TextBlob polarity + subjectivity
#    - Custom financial lexicon
#  [COMPARISON] Test on sample sentences

class LexiconSentimentAnalyzer:
    def vader_sentiment(text)          # → compound, label
    def textblob_sentiment(text)       # → polarity, label
    def custom_lexicon_sentiment(text) # → score
    def analyze(text)                  # → all three + ensemble

# Expected output:
{
    'text': 'Excellent earnings beat expectations',
    'vader': {'compound': 0.82, 'label': 'positive'},
    'textblob': {'polarity': 0.75, 'label': 'positive'},
    'custom_score': 0.80,
    'ensemble_score': 0.79,
    'confidence': 0.82
}
```

**Acceptance Criteria:**
- VADER working correctly
- TextBlob polarity in [0, 1] range
- Custom lexicon with >20 words
- Ensemble averaging correctly
- Handles edge cases
- Consistent results on repeated runs

---

### Notebook 3: task_3_evaluation.ipynb

```python
#  [LOAD] Import Financial PhraseBank (4,844 sentences)
#  [PREPROCESS] Apply Task 1 preprocessing
#  [SENTIMENT] Apply Task 2 sentiment scoring
#  [EVALUATE] Calculate metrics
#    - Overall accuracy
#    - Per-class precision, recall, F1
#    - Confusion matrix
#  [VISUALIZE] Create plots
#    - Confusion matrix heatmap
#    - F1-score bar chart

# Expected output:
# Accuracy: 0.75-0.78 (typical)
# Positive F1: 0.78
# Negative F1: 0.78
# Neutral F1: 0.69
```

---
