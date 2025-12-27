# Week 3: Transformers & Fine-tuned FinBERT for Sentiment Analysis

In Week 3, you'll use transfer learning to build a state-of-the-art sentiment analysis system using **FinBERT**, a pre-trained transformer model specifically designed for financial text. This week answers the question: **"Can deep learning transformers beat supervised ML from Week 2?"**

**What you'll build:**
- Understanding of transformer architecture and pre-training
- FinBERT-based sentiment analyzer with zero-shot inference
- Fine-tuned FinBERT model on Financial PhraseBank
- Comprehensive 5-model comparison (VADER + 3 ML models + FinBERT)

---

## Three Tasks This Week

### Task 1: Transformer Basics & FinBERT Exploration

**What:** Understand how transformers work and load pre-trained FinBERT model

**Learn:** 
- Transformer architecture (attention mechanism, tokenization, embeddings)
- What pre-training means and why it matters
- How to use Hugging Face transformers library
- FinBERT model specifics (trained on financial news, 10M sentences)

**Output:** FinBERT model loaded and tested on sample sentences

**Example:**
```python
# Load pre-trained FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Analyze sentiment of financial text
text = "Revenue increased significantly, beating analyst expectations"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
probabilities = outputs.logits.softmax(dim=-1)

# Output: 
# Negative: 0.02, Neutral: 0.15, Positive: 0.83
```

**Why FinBERT is powerful:**
- Pre-trained on 10 million financial news sentences
- Understands financial vocabulary ("beat", "miss", "guidance")
- Can capture complex context ("not good" vs "good")
- Attention mechanism focuses on relevant words

---

### Task 2: Fine-tune FinBERT on Financial PhraseBank

**What:** Adapt FinBERT specifically to your dataset through fine-tuning

**Learn:**
- Transfer learning and fine-tuning concepts
- Training loop with PyTorch/HuggingFace
- Learning rate scheduling and early stopping
- Validation during training to prevent overfitting

**Output:** Custom FinBERT model trained on Financial PhraseBank

**Example:**
```python
# Fine-tuning process
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./finbert_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
# Output: Model improves from 0.83 → 0.87 F1 after fine-tuning
```

**Question answered:** Does fine-tuning FinBERT beat zero-shot FinBERT and Week 2 ML?

---

### Task 3: Model Comparison & Final Analysis

**What:** Compare all five sentiment analysis approaches and draw conclusions

**Learn:**
- Comprehensive benchmarking methodology
- Trade-offs between accuracy, speed, and complexity
- Statistical significance testing
- Choosing the right model for production

**Output:** 
- Comparison table of all 5 models
- ROC curves for each approach
- Performance vs speed/complexity scatter plot
- Final recommendation with trade-off analysis

**Example:**
```
COMPREHENSIVE MODEL COMPARISON
═════════════════════════════════════════════════════════════════

Model                          Accuracy  F1-Score  Speed       GPU?
─────────────────────────────────────────────────────────────────
Week 1: VADER (Lexicon)        0.75      0.75      <1ms        No
Week 2: Logistic Regression    0.80      0.80      <1ms        No
Week 2: Naive Bayes            0.79      0.78      <1ms        No
Week 2: KNN                    0.76      0.75      10ms        No
Week 3: FinBERT (zero-shot)    0.83      0.83      100ms       Yes
Week 3: FinBERT (fine-tuned)   0.87      0.87      100ms       Yes

Key Insight:
- FinBERT (fine-tuned) achieves 0.87 F1 (+0.07 vs Week 2 best)
- Trade-off: 100× slower than ML but 3% more accurate
- Question for Week 4-5: Worth the slowdown on real trading data?
```

---

## Resources & Learning Materials

### Must-Read Articles 

1. **Attention Is All You Need**
   - https://towardsdatascience.com/attention-is-all-you-need-explained-207ebc2dc19b
   - Read: What transformers are, why attention matters, intuition

2. **BERT Explained: State of the art language model for NLP**
   - https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6fb0
   - Read: How BERT is different from transformers, pre-training, fine-tuning

3. **Fine-tuning BERT for Text Classification**
   - https://towardsdatascience.com/sentiment-analysis-with-bert-transformers-and-tensorflow-5f5b16df2a72
   - Read: Step-by-step fine-tuning process, best practices

### Must-Watch Videos 

1. **Transformers Explained** 
   - https://www.youtube.com/watch?v=TQQlZhbC5ps 
   - Watch: Attention mechanism, why transformers revolutionized NLP

2. **BERT Explained** 
   - YouTube: "BERT explained StatQuest"
   - Watch: Bidirectional training, masked language modeling

3. **Hugging Face Tutorial: Fine-tuning BERT** 
   - YouTube: "Hugging Face fine-tuning tutorial"
   - Watch: Practical implementation, training loop

4. **Transfer Learning in NLP** 
   - YouTube: "Transfer learning explained"
   - Watch: Why pre-training helps, fine-tuning vs training from scratch

### Reference Documentation

- **Hugging Face Transformers:** https://huggingface.co/transformers/
- **FinBERT Model Card:** https://huggingface.co/ProsusAI/finbert
- **PyTorch:** https://pytorch.org/
- **scikit-learn Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Dataset

### Financial PhraseBank (Same as Week 1 & 2)
---

## Three Jupyter Notebooks to Submit

### Notebook 1: task_1_transformer_basics.ipynb

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

class FinBERTAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Load pre-trained FinBERT."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def analyze_sentiment_zero_shot(self, text):
        """
        Get sentiment prediction without fine-tuning.
        Returns: {predicted_label, confidence, probabilities}
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'text': text,
            'predicted_label': self.label_map[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            }
        }
    
    def analyze_batch(self, texts):
        """Efficiently analyze multiple texts."""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment_zero_shot(text))
        return results
```

**Acceptance Criteria:**
- FinBERT model loads successfully
- Zero-shot inference works on sample sentences
- Output format includes confidence scores
- Batch processing implemented
- Code runs on Google Colab (GPU or CPU)
- Comparison to Week 1-2 models documented

**Expected Output:**
```
FINBERT ZERO-SHOT ANALYSIS
═════════════════════════════════════════════════════════════

Sample 1: "Strong earnings growth beat expectations"
  Predicted: positive (confidence: 0.91)
  Probabilities: negative=0.02, neutral=0.07, positive=0.91

Sample 2: "Revenue declined significantly"
  Predicted: negative (confidence: 0.88)
  Probabilities: negative=0.88, neutral=0.10, positive=0.02

Sample 3: "Results were in line with guidance"
  Predicted: neutral (confidence: 0.72)
  Probabilities: negative=0.10, neutral=0.72, positive=0.18

Comparison to Week 2 (zero-shot vs ML models):
  FinBERT zero-shot: No training needed, but needs GPU
  Logistic Regression: Fast, no GPU, but needs training
  Trade-off: Accuracy vs speed vs resources
```

---

### Notebook 2: task_2_finetune_finbert.ipynb

```python

from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

class FinBERTFineTuner:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Initialize for fine-tuning."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def prepare_dataset(self, texts, labels, max_length=128):
        """
        Prepare texts for training.
        FinBERT expects: [text, label]
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        dataset = Dataset.from_dict({
            'text': texts,
            'label': labels
        })
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset
    
    def fine_tune(self, train_texts, train_labels, val_texts, val_labels, 
                  num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Fine-tune FinBERT on your data.
        
        Parameters:
        - num_epochs: 3 is typical (more overfits, less undertains)
        - batch_size: 16 for Colab GPU, 8 if memory limited
        - learning_rate: 2e-5 is standard for fine-tuning
        """
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./finbert_finetuned',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
        )
        
        # Define metrics
        def compute_metrics(eval_preds):
            predictions, labels = eval_preds
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("Starting fine-tuning...")
        trainer.train()
        print("Fine-tuning complete!")
        
        return trainer
```

**Acceptance Criteria:**
- Fine-tuning code completes without errors
- Training and validation losses decrease
- Final validation accuracy ≥0.85
- Model checkpoint saved
- Training curves plotted
- Comparison: zero-shot vs fine-tuned performance

**Expected Output:**
```
FINBERT FINE-TUNING RESULTS
═════════════════════════════════════════════════════════════

Training completed in 45 minutes (on Colab GPU)

VALIDATION PERFORMANCE BY EPOCH:
Epoch 1: Loss=0.45, Accuracy=0.82, F1=0.81
Epoch 2: Loss=0.38, Accuracy=0.85, F1=0.85
Epoch 3: Loss=0.35, Accuracy=0.87, F1=0.87 ← Best

COMPARISON: ZERO-SHOT vs FINE-TUNED
─────────────────────────────────────────
Model                  Accuracy  F1-Score
─────────────────────────────────────────
FinBERT Zero-shot      0.83      0.83
FinBERT Fine-tuned     0.87      0.87
Improvement            +0.04     +0.04

INTERPRETATION:
- Fine-tuning improves F1 from 0.83 → 0.87 (+4%)
- Model learns dataset-specific patterns
- Better than any ML model (0.80 F1 from Week 2)
- Trade-off: 45 min training vs instant inference (once trained)
```

---

### Notebook 3: task_3_comprehensive_comparison.ipynb

```python

class ComprehensiveComparison:
    def __init__(self):
        """Load all 5 models."""
        self.models = {}
        self.results = {}
    
    def load_all_models(self, 
                        week1_preprocessor,  # From Week 1
                        week2_feature_extractor,  # From Week 2
                        week2_lr_model,  # From Week 2
                        week2_nb_model,  # From Week 2
                        week2_knn_model,  # From Week 2
                        finbert_model):  # From Week 3
        """Load all trained models."""
        self.models['VADER'] = week1_preprocessor
        self.models['TF-IDF + LogisticRegression'] = (week2_feature_extractor, week2_lr_model)
        self.models['TF-IDF + NaiveBayes'] = (week2_feature_extractor, week2_nb_model)
        self.models['TF-IDF + KNN'] = (week2_feature_extractor, week2_knn_model)
        self.models['FinBERT'] = finbert_model
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate each model on same test set.
        Returns: {model_name: metrics}
        """
        
        # VADER (Week 1)
        vader_preds = self._predict_vader(X_test)
        vader_metrics = self._calculate_metrics(y_test, vader_preds)
        
        # Logistic Regression (Week 2)
        lr_preds = week2_lr_model.predict(X_test)
        lr_metrics = self._calculate_metrics(y_test, lr_preds)
        
        # ... (similar for NB, KNN)
        
        # FinBERT (Week 3)
        finbert_preds = self._predict_finbert(X_test)
        finbert_metrics = self._calculate_metrics(y_test, finbert_preds)
        
        return {
            'VADER': vader_metrics,
            'Logistic Regression': lr_metrics,
            'FinBERT': finbert_metrics,
            # ... etc
        }
    
    def compare_and_visualize(self, results):
        """Create comparison table and plots."""
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        
        # Plot 1: Accuracy comparison
        plt.figure(figsize=(10, 6))
        comparison_df['accuracy'].plot(kind='barh')
        plt.xlabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        
        # Plot 2: ROC curves for each model
        # (one plot with 5 lines, one for each model)
        
        return comparison_df
    
    def trade_off_analysis(self, results):
        """
        Analyze accuracy vs speed vs complexity.
        Write 500+ word analysis.
        """
        # Speed comparison
        speeds = {
            'VADER': '<1ms',
            'Logistic Regression': '<1ms',
            'KNN': '10ms',
            'FinBERT': '100ms'
        }
        
        # Complexity (learning curve)
        complexity = {
            'VADER': 'Simple (lexicon)',
            'Logistic Regression': 'Medium (ML)',
            'FinBERT': 'High (transformers)'
        }
        
        # Accuracy
        accuracy = comparison_df['f1'].to_dict()
        
        # Output: discussion of trade-offs
```

**Acceptance Criteria:**
- All 5 models evaluated on same test set
- Comparison table created (accuracy, precision, recall, F1)
- ROC curves plotted for each model
- Speed comparison documented
- Confusion matrices visualized (heatmaps)

**Expected Output:**
```
COMPREHENSIVE 5-MODEL COMPARISON
═════════════════════════════════════════════════════════════════════

Model                          Accuracy  Precision  Recall    F1-Score
───────────────────────────────────────────────────────────────────────
Week 1: VADER                  0.75      0.76       0.74      0.75
Week 2: Logistic Regression    0.80      0.82       0.79      0.80
Week 2: Naive Bayes            0.79      0.80       0.77      0.78
Week 2: KNN (k=5)              0.76      0.77       0.75      0.76
Week 3: FinBERT (fine-tuned)   0.87      0.88       0.86      0.87

RANKING BY F1-SCORE:
1. FinBERT (0.87) ← Best
2. Logistic Regression (0.80)
3. Naive Bayes (0.78)
4. VADER (0.75)
5. KNN (0.76)

TRADE-OFF ANALYSIS:
─────────────────────────────────────────────────────────────────────
                    Accuracy  Speed    Complexity  GPU Needed
─────────────────────────────────────────────────────────────────────
VADER               0.75      <1ms     Low         No
Logistic Regression 0.80      <1ms     Medium      No
FinBERT             0.87      100ms    High        Yes (but colab OK)

INTERPRETATION:
- FinBERT is 12% better than VADER (0.87 vs 0.75)
- FinBERT is 9% better than best ML (0.87 vs 0.80)
- Trade-off: 100× slower but 9% more accurate
- Question for Week 4-5: Is 9% improvement worth slowdown on real data?
- Answer: Week 4 will test on real stock prices + news
           Week 5 will backtest which actually makes money

RECOMMENDATIONS:
- For accuracy-critical tasks: Use FinBERT (0.87)
- For speed-critical tasks: Use Logistic Regression (0.80)
- For simple baseline: VADER still useful for quick checks
```

---

## Setup for Google Colab

```python
# Cell 1: Enable GPU
# Runtime → Change runtime type → GPU (T4)

# Cell 2: Install packages
!pip install transformers torch huggingface-hub datasets

# Cell 3: Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 4: Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Cell 5: Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✓ Setup complete! Ready for Week 3")
```

---
## Key Insights for Week 3

### Why Transformers Work Better

Transformers solve limitations of earlier methods through:

1. **Context Understanding:** BERT reads entire sentence, not just word counts
   - Example: "not good" vs "good"
   - BoW sees: [not, good] for both
   - BERT sees: different contexts, handles negation

2. **Attention Mechanism:** Model focuses on relevant words
   - Example: "Company X beat earnings estimates on strong guidance"
   - BERT focuses on: "beat", "strong" (relevant)
   - Ignores: "on", "earnings" (less important)

3. **Pre-training:** Learned from 10M financial documents
   - Understands financial vocabulary before fine-tuning
   - Zero-shot already achieves 0.83 F1
   - Fine-tuning adapts to your specific dataset

4. **Bidirectional Learning:** Reads context from both directions
   - Earlier models (like old LSTM) only look backwards
   - BERT looks forward and backward simultaneously
   - Better understanding of relationships

### When to Use Each Approach

```
Use VADER (Week 1) when:
- Speed is critical (<1ms requirement)
- No training data available
- Need interpretability
- Baseline check
Example: Real-time monitoring of thousands of tweets

Use Logistic Regression (Week 2) when:
- Need balance of accuracy and speed
- Have training data (100+ labeled examples)
- Want some interpretability (feature weights)
- Limited computational resources
Example: Mobile app inference

Use FinBERT (Week 3) when:
- Accuracy is most important
- Have access to GPU
- Can tolerate 100ms latency
- Have training data (hundreds of examples)
- Domain-specific fine-tuning helps
Example: High-stakes financial decision making
```
---
