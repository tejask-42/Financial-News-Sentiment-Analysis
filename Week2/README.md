# Week 2: Supervised Learning

## Overview

By the end of Week 2, you will:
- Understand Bag-of-Words and TF-IDF feature extraction
- Train and compare three ML models
- Perform hyperparameter tuning with GridSearchCV
- Analyze precision-recall tradeoffs and ROC curves
- **Beat Week 1's 0.75 F1 baseline with ~0.80-0.82 F1**

---

## Resources

### Useful Articles

1. **Bag of Words Explained**
   - Link: https://towardsdatascience.com/bag-of-words-explained-9f29f985cbaf
   - Read: Why BoW works, limitations, comparison with other methods

2. **TF-IDF: What It Is and Why It Matters**
   - Link: https://www.analyticsvidhya.com/blog/2020/08/tfidf-explained/
   - Read: TF-IDF formula, why better than BoW, worked examples

3. **Logistic Regression for Text Classification**
   - Link: https://towardsdatascience.com/logistic-regression-explained-9ee5cdf8b4f2
   - Read: How LR works, interpretability, when to use

### Useful Videos 

1. **Bag of Words & TF-IDF Explained** 
   - Link: https://www.youtube.com/watch?v=D8JpPnH-aVs 
   - Watch: Visual explanation, comparison

2. **Logistic Regression Explained** 
   - YouTube: "Logistic Regression explained StatQuest"
   - Watch: How it differs from linear regression

3. **K-Nearest Neighbors Explained** 
   - YouTube: "KNN explained StatQuest"
   - Watch: Distance metrics, why "lazy learning"

4. **Naive Bayes for Classification** 
   - YouTube: "Naive Bayes explained StatQuest"
   - Watch: Probabilistic approach, independence assumption

### Reference Docs

- **scikit-learn CountVectorizer:** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- **scikit-learn TfidfVectorizer:** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- **scikit-learn GridSearchCV:** https://scikit-learn.org/stable/modules/model_selection.html#grid-search
- **scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Dataset

### Financial PhraseBank (Same as Week 1)

---

## Three Tasks

### Task 1: Feature Extraction 

**Deliverable:** `task_1_feature_extraction.ipynb`

#### Objective
Build a `FeatureExtractor` class that converts text to numerical features using both Bag-of-Words and TF-IDF.

#### Class Structure

```python
class FeatureExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize vectorizers.
        
        Parameters:
        - max_features: Keep top 5000 most frequent words
        - ngram_range: (1,2) means unigrams + bigrams
          Example: "profit increase" → 
            unigrams: ['profit', 'increase']
            bigrams: ['profit increase']
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,           # Ignore words in <2 docs
            max_df=0.8,         # Ignore words in >80% docs
            stop_words='english'
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
    
    def fit_transform_bow(self, texts):
        """
        Create Bag-of-Words feature matrix.
        Returns: (sparse matrix, feature names)
        """
        pass
    
    def fit_transform_tfidf(self, texts):
        """
        Create TF-IDF feature matrix.
        Returns: (sparse matrix, feature names)
        """
        pass
    
    def get_top_features(self, vectorizer, X, n=20):
        """
        Get top n most important features by TF-IDF score.
        Returns: [(feature_name, score), ...]
        """
        pass
    
    def visualize_top_features(self, vectorizer, X, n=20):
        """
        Plot top features as horizontal bar chart.
        Returns: matplotlib figure
        """
        pass
```

#### Acceptance Criteria

-  BoW creation works (sparse matrix)
-  TF-IDF creation works (sparse matrix)
-  Unigrams + bigrams included
-  Top 20 features visualized
-  Feature names are strings

#### Expected Output

```
FEATURE EXTRACTION RESULTS
═══════════════════════════════════════════════════════════

BoW Shape: (4844, 5000)
TF-IDF Shape: (4844, 5000)
Vocabulary Size: 5000

Top 10 Most Informative Features (by average TF-IDF):
1. profit          (avg_tfidf: 0.95)
2. revenue         (avg_tfidf: 0.92)
3. loss            (avg_tfidf: 0.91)
4. earnings        (avg_tfidf: 0.89)
5. growth          (avg_tfidf: 0.87)
6. increase        (avg_tfidf: 0.85)
7. strong          (avg_tfidf: 0.84)
8. decline         (avg_tfidf: 0.83)
9. beat            (avg_tfidf: 0.82)
10. miss           (avg_tfidf: 0.80)

Sample: "strong profit growth"
  BoW features: [strong=1, profit=1, growth=1]
  TF-IDF features: [strong=0.45, profit=0.48, growth=0.42]
```


### Task 2: Train & Compare Classification Models 

**Deliverable:** `task_2_classification_models.ipynb`

#### Objective
Train three ML models, perform hyperparameter tuning, and compare performance.

#### Class Structure

```python
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """
        Train Logistic Regression with hyperparameter tuning.
        
        Hyperparameters to tune:
        - C: Regularization strength [0.1, 1.0, 10.0]
          C < 1: More regularization (simpler model)
          C > 1: Less regularization (complex model)
        
        Returns: (trained_model, metrics_dict)
        """
        param_grid = {'C': [0.1, 1.0, 10.0]}
        lr = LogisticRegression(max_iter=1000, random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, y_pred)
        self.models['Logistic Regression'] = best_model
        self.results['Logistic Regression'] = metrics
        
        return best_model, metrics
    
    def train_naive_bayes(self, X_train, y_train, X_val, y_val):
        """
        Train Naive Bayes (no hyperparameter tuning needed).
        Works well with sparse text features.
        
        Returns: (trained_model, metrics_dict)
        """
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, y_pred)
        self.models['Naive Bayes'] = nb
        self.results['Naive Bayes'] = metrics
        
        return nb, metrics
    
    def train_knn(self, X_train, y_train, X_val, y_val):
        """
        Train K-Nearest Neighbors with hyperparameter tuning.
        
        Hyperparameters to tune:
        - n_neighbors: [3, 5, 7]
          k=3: More complex, follows data closely
          k=7: Simpler, more stable
        - metric: 'cosine' (better for sparse text)
        
        Returns: (trained_model, metrics_dict)
        """
        param_grid = {'n_neighbors': [3, 5, 7]}
        knn = KNeighborsClassifier(metric='cosine')
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        
        metrics = self._calculate_metrics(y_val, y_pred)
        self.models['KNN'] = best_model
        self.results['KNN'] = metrics
        
        return best_model, metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def compare_models(self):
        """Create comparison table and visualizations."""
        df_results = pd.DataFrame(self.results).T
        return df_results
```

#### Acceptance Criteria

- Logistic Regression trains with GridSearchCV
- Naive Bayes trains successfully
- KNN trains with hyperparameter tuning
- 5-fold cross-validation used (at least for GridSearchCV)
- Comparison table generated
- Best model identified with justification


#### Expected Output

```
MODEL COMPARISON ON VALIDATION SET (15% of data)
═══════════════════════════════════════════════════════════════

                      Accuracy  Precision  Recall   F1-Score
─────────────────────────────────────────────────────────────
Logistic Regression    0.82      0.84       0.80     0.82
Naive Bayes           0.79      0.81       0.76     0.78
KNN (k=5)             0.76      0.78       0.73     0.75

Best Model: Logistic Regression
  - Highest accuracy: 0.82
  - Balanced precision/recall
  - Not overfit (not too different from other models)

Comparison to Week 1:
  Week 1 (Lexicon):     F1 = 0.75
  Week 2 (LR best):     F1 = 0.82
  Improvement:          +0.07 (+9.3%) ✓
```


### Task 3: Model Evaluation & Hyperparameter Tuning 

**Deliverable:** `task_3_model_evaluation.ipynb`

#### Objective
Thoroughly evaluate best model on test set and analyze failure cases.

#### Key Analyses

**1. Test Set Evaluation**
```python
# Use model from Task 2 (trained on 70% data)
# Evaluate on 15% test set (NEVER USED FOR TRAINING)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_per_class = precision_score(y_test, y_pred, average=None)
recall_per_class = recall_score(y_test, y_pred, average=None)
f1_per_class = f1_score(y_test, y_pred, average=None)
cm = confusion_matrix(y_test, y_pred)
```

**2. Cross-Validation**
```python
# 5-fold CV on training set to check model stability
from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
# Good if std is small (<0.05), meaning stable across folds
```

**3. ROC-AUC Curve**
```python
# For each class (one-vs-rest)
for class_label in [0, 1, 2]:  # positive, negative, neutral
    y_true_binary = (y_test == class_label).astype(int)
    y_pred_proba = best_model.predict_proba(X_test)[:, class_label]
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
    roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
    
    plt.plot(fpr, tpr, label=f'Class {class_label} (AUC={roc_auc:.3f})')
```

**4. Precision-Recall Tradeoff**
```python
# For best class (usually positive or negative)
from sklearn.metrics import precision_recall_curve

y_true_binary = (y_test == 1).astype(int)  # Example: positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)

plt.plot(recall, precision)
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
# Shows tradeoff: can increase recall by lowering threshold, but precision drops
```

**5. Failure Case Analysis**
```python
# Find misclassified examples
misclassified_indices = np.where(y_pred != y_test)[0]

# Get top 5 with highest confidence (but wrong prediction)
top_misclassified = misclassified_indices[:5]

for idx in top_misclassified:
    text = get_original_text(idx)  # Need to keep original text
    predicted = y_pred[idx]
    actual = y_test[idx]
    confidence = max(best_model.predict_proba(X_test)[idx])
    
    print(f"Text: {text}")
    print(f"Predicted: {predicted}, Actual: {actual}, Confidence: {confidence:.2f}")
    print(f"Why did model fail? [Your analysis]")
```

#### Acceptance Criteria

- Test accuracy ≥0.75 (target ≥0.80)
- Per-class metrics reported (positive, negative, neutral)
- ROC curve plotted (should be above diagonal line)
- ROC-AUC ≥0.80
- Precision-Recall curve plotted
- 5-fold cross-validation results reported (mean ± std)
- Top 5 misclassified examples identified
- Confusion matrix visualized (heatmap)


#### Expected Output

```
TEST SET EVALUATION (Final Performance on Unseen Data)
═══════════════════════════════════════════════════════════════

Model: Logistic Regression (best from Task 2)

OVERALL METRICS:
  Accuracy: 0.80
  Weighted F1: 0.80
  
PER-CLASS METRICS:
  Positive:
    Precision: 0.84    (of predicted positive, 84% correct)
    Recall: 0.80       (of actual positive, caught 80%)
    F1-Score: 0.82
    
  Negative:
    Precision: 0.82
    Recall: 0.78
    F1-Score: 0.80
    
  Neutral:
    Precision: 0.76
    Recall: 0.74
    F1-Score: 0.75

CONFUSION MATRIX:
          Predicted+  Predicted-  Predicted0
Actual+      1000        95         105
Actual-       110       1050        140
Actual0       180       200         920

5-FOLD CROSS-VALIDATION:
  Scores: [0.81, 0.80, 0.79, 0.82, 0.80]
  Mean: 0.804 ± 0.010
  → Model is stable across different data splits

ROC-AUC SCORES:
  Positive class: 0.85
  Negative class: 0.83
  Neutral class: 0.78
  → All >0.75, so model has good discriminative power

COMPARISON TO WEEK 1:
  Week 1 (Lexicon VADER):     F1 = 0.75
  Week 2 (Logistic Regression): F1 = 0.80
  Improvement:                +0.05 (+6.7%)
```

---
