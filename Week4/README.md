# Week 4: Real Data Integration & Stock Price Correlation Analysis

In Week 4, you'll test your sentiment analysis systems on real financial data. This week answers the critical question: **"Do the lab results from Weeks 1-3 actually predict real stock prices?"**

You'll work with real stock prices from yfinance and financial news from Kaggle, correlating sentiment scores with stock price movements to discover which approach (lexicon, ML, or transformers) best predicts market behavior.

**What you'll build:**
- Real-time stock price data pipeline (yfinance integration)
- Financial news data collection and preprocessing
- Sentiment scoring of real news using all 3 methods (VADER, ML, FinBERT)
- Time-aligned sentiment and price correlation analysis
- Visualization of sentiment-price relationships and trading signals

---

## Three Tasks This Week

### Task 1: Real Data Collection & Preprocessing

**What:** Load real stock prices and financial news, align them by date

**Learn:**
- How to fetch historical stock data using yfinance
- How to load financial news datasets (Kaggle)
- Time-series alignment and handling missing data
- Data quality checks and validation

**Output:** Synchronized dataset with dates, stock prices, and news headlines

**Example:**
```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Download 6 months of stock price data
ticker = "AAPL"
start_date = "2024-01-01"
end_date = "2024-06-30"

df_prices = yf.download(ticker, start=start_date, end=end_date)
print(df_prices.head())

# Output:
#            Open   High    Low  Close  Volume
# Date
# 2024-01-01 NaN    NaN     NaN   NaN    0
# 2024-01-02 188.5  190.2   188.1 189.9 52345600

# Load financial news (Kaggle dataset)
df_news = pd.read_csv('financial_news.csv')
# Expected columns: date, headline, ticker, sentiment_label (optional)

# Align news and prices by date
df_combined = pd.merge(
    df_prices.reset_index().rename(columns={'Date': 'date'}),
    df_news,
    on='date',
    how='inner'
)

print(f"Combined dataset shape: {df_combined.shape}")
# Shape: (127 days, 8 columns)
```
---

### Task 2: Apply All 3 Sentiment Methods to Real News

**What:** Score the same real news headlines using VADER, Logistic Regression, and FinBERT

**Learn:**
- How to load and apply models trained in Weeks 1-3
- Batch processing for efficiency
- Handling different output formats from different models
- Creating consistent sentiment scores across models

**Output:** DataFrame with sentiment scores from all 3 methods for each news headline

**Example:**
```python
# Load all 3 sentiment models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Method 1: VADER (from Week 1)
from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

# Method 2: Logistic Regression (from Week 2)
from sklearn.feature_extraction.text import TfidfVectorizer
lr_model = load_model_from_week2()

# Method 3: FinBERT (from Week 3)
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Apply to real news
results = []
for headline in df_combined['headline']:
    # VADER
    vader_score = vader.polarity_scores(headline)['compound']
    
    # Logistic Regression
    tfidf_features = vectorizer.transform([headline])
    lr_pred_proba = lr_model.predict_proba(tfidf_features)[0]
    lr_sentiment = lr_pred_proba[2] - lr_pred_proba[0]  # positive - negative
    
    # FinBERT
    inputs = finbert_tokenizer(headline, return_tensors="pt", truncation=True)
    outputs = finbert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    finbert_sentiment = probs[2] - probs[0]  # positive - negative
    
    results.append({
        'headline': headline,
        'date': headline_date,
        'vader_sentiment': vader_score,
        'lr_sentiment': lr_sentiment,
        'finbert_sentiment': finbert_sentiment
    })

df_sentiments = pd.DataFrame(results)

# Example output:
#                            headline       date  vader_sentiment  lr_sentiment  finbert_sentiment
# 0  Apple beats earnings expectations 2024-01-15         0.72          0.65              0.78
# 1  Tech stocks decline amid uncertainty 2024-01-16        -0.45         -0.52             -0.58
# 2  Market remains stable with profit gains 2024-01-17     0.35          0.28              0.42
```

**Question answered:** Do all 3 methods agree on sentiment, or do they differ?

Expected insight: FinBERT is more confident, ML is moderate, VADER is simpler.

---

### Task 3: Correlation Analysis & Trading Signal Generation

**What:** Correlate sentiment with price changes and identify predictive signals

**Learn:**
- How to calculate price changes (returns)
- Correlation analysis and statistical testing
- Rolling window correlations
- Feature importance and predictive power

**Output:**
- Correlation table showing which sentiment predicts prices best
- Time-series plots of sentiment vs returns
- Trading signals generated from each sentiment method
- Statistical significance tests

**Example:**
```python
import numpy as np
from scipy import stats

# Calculate price returns
df_combined['price_change'] = df_combined['Close'].pct_change()
df_combined['next_day_return'] = df_combined['price_change'].shift(-1)

# Calculate correlations
corr_vader = df_combined['vader_sentiment'].corr(df_combined['next_day_return'])
corr_lr = df_combined['lr_sentiment'].corr(df_combined['next_day_return'])
corr_finbert = df_combined['finbert_sentiment'].corr(df_combined['next_day_return'])

# Statistical significance testing
r_vader, p_vader = stats.pearsonr(df_combined['vader_sentiment'], 
                                   df_combined['next_day_return'])
r_lr, p_lr = stats.pearsonr(df_combined['lr_sentiment'], 
                             df_combined['next_day_return'])
r_finbert, p_finbert = stats.pearsonr(df_combined['finbert_sentiment'], 
                                       df_combined['next_day_return'])

print("SENTIMENT-PRICE CORRELATION ANALYSIS")
print("="*60)
print(f"VADER:      r={r_vader:.3f}, p-value={p_vader:.4f}")
print(f"Logistic Regression: r={r_lr:.3f}, p-value={p_lr:.4f}")
print(f"FinBERT:    r={r_finbert:.3f}, p-value={p_finbert:.4f}")

# Example output:
# SENTIMENT-PRICE CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════
# VADER:      r=0.32, p-value=0.0012
# Logistic Regression: r=0.38, p-value=0.0001
# FinBERT:    r=0.42, p-value<0.0001

# Generate trading signals
def generate_signal(sentiment_score, threshold=0.3):
    if sentiment_score > threshold:
        return 1  # Buy signal
    elif sentiment_score < -threshold:
        return -1  # Sell signal
    else:
        return 0  # Hold

df_combined['signal_vader'] = df_combined['vader_sentiment'].apply(
    lambda x: generate_signal(x, threshold=0.3)
)
df_combined['signal_lr'] = df_combined['lr_sentiment'].apply(
    lambda x: generate_signal(x, threshold=0.3)
)
df_combined['signal_finbert'] = df_combined['finbert_sentiment'].apply(
    lambda x: generate_signal(x, threshold=0.3)
)

# Count signals
print("\nTRADING SIGNALS GENERATED")
print("="*60)
print(f"Buy signals (VADER):       {(df_combined['signal_vader'] == 1).sum()}")
print(f"Sell signals (VADER):      {(df_combined['signal_vader'] == -1).sum()}")
print(f"Buy signals (Logistic Reg):{(df_combined['signal_lr'] == 1).sum()}")
print(f"Buy signals (FinBERT):     {(df_combined['signal_finbert'] == 1).sum()}")
```

**Question answered:** Which sentiment method best predicts next-day returns?

---

## Resources & Learning Materials

### Must-Read Articles

1. **Introduction to yfinance: Fetching Financial Data**
   - https://towardsdatascience.com/fetching-stock-data-using-python-a-guide-to-yfinance-library-90b2b50a6c62
   - Read: How to download stock data, handle dates, data quality

2. **Correlation Analysis in Finance**
   - https://www.investopedia.com/terms/c/correlation.asp
   - Read: Understanding correlation, interpreting correlation coefficients

3. **Sentiment Analysis for Trading: From Theory to Practice**
   - https://towardsdatascience.com/stock-market-sentiment-analysis-with-lstm-neural-networks-92ebde6ce755
   - Read: Real-world challenges, sentiment-price relationships

### Must-Watch Videos

1. **yfinance Tutorial: Download Stock Data**
   - https://www.youtube.com/watch?v=5yfh5OYhLCA
   - Watch: Download prices, handle missing data, time-series basics

2. **Correlation and Causation in Finance**
   - YouTube: "Correlation vs causation explained"
   - Watch: Why correlation doesn't mean causation, pitfalls

3. **Feature Engineering for Time Series**
   - YouTube: "Time series feature engineering Python"
   - Watch: Lagged features, rolling windows, feature creation

4. **Statistical Testing in Python**
   - YouTube: "SciPy statistics tutorial"
   - Watch: Correlation tests, p-values, significance

### Reference Documentation

- **yfinance Documentation:** https://yfinance.readthedocs.io/
- **Pandas Time Series:** https://pandas.pydata.org/docs/user_guide/timeseries.html
- **SciPy Stats:** https://docs.scipy.org/doc/scipy/reference/stats.html
- **Kaggle Financial News Datasets:** https://www.kaggle.com/search?q=financial+news

---

## Datasets

### Stock Price Data (yfinance)

**Source:** Free, real-time data from Yahoo Finance via yfinance  
**Coverage:** Any ticker (AAPL, GOOGL, MSFT, etc.)  
**Format:** Daily OHLCV (Open, High, Low, Close, Volume)  
**Period:** Choose any date range (recommend 6+ months)  

```python
import yfinance as yf
df = yf.download("AAPL", start="2024-01-01", end="2024-06-30")
```

### Financial News Data (Kaggle)

**Source:** https://www.kaggle.com/datasets/jeet2016/us-stock-market-news-data  
**Alternative:** https://www.kaggle.com/datasets/hadoopadmin/financial-news-dataset  

---

## Three Jupyter Notebooks to Submit

### Notebook 1: task_1_real_data_collection.ipynb

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class RealDataPipeline:
    def __init__(self, ticker="AAPL", start_date="2024-01-01", end_date="2024-06-30"):
        """Initialize data pipeline."""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df_prices = None
        self.df_news = None
        self.df_combined = None
    
    def download_stock_prices(self):
        """
        Download historical stock prices using yfinance.
        Returns: DataFrame with OHLCV data
        """
        print(f"Downloading {self.ticker} prices from {self.start_date} to {self.end_date}...")
        self.df_prices = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Reset index to make Date a column
        self.df_prices = self.df_prices.reset_index()
        self.df_prices.rename(columns={'Date': 'date'}, inplace=True)
        self.df_prices['date'] = pd.to_datetime(self.df_prices['date']).dt.date
        
        print(f"Downloaded {len(self.df_prices)} days of price data")
        return self.df_prices
    
    def load_financial_news(self, news_csv_path):
        """
        Load financial news dataset from CSV.
        Expected columns: date, headline, (optional) ticker
        """
        print(f"Loading news data from {news_csv_path}...")
        self.df_news = pd.read_csv(news_csv_path)
        
        # Convert date to datetime and extract date only
        self.df_news['date'] = pd.to_datetime(self.df_news['date']).dt.date
        
        # Filter by ticker if column exists
        if 'ticker' in self.df_news.columns:
            self.df_news = self.df_news[self.df_news['ticker'] == self.ticker]
        
        print(f"Loaded {len(self.df_news)} news articles for {self.ticker}")
        return self.df_news
    
    def align_data(self):
        """
        Align news and prices by date.
        Aggregate multiple news articles per day.
        """
        # Group news by date (multiple headlines per day)
        df_news_daily = self.df_news.groupby('date')['headline'].apply(
            lambda x: ' '.join(x)  # Combine headlines for same day
        ).reset_index()
        df_news_daily.rename(columns={'headline': 'all_headlines'}, inplace=True)
        
        # Merge prices and news
        self.df_combined = self.df_prices.merge(
            df_news_daily,
            on='date',
            how='inner'
        )
        
        print(f"Combined dataset shape: {self.df_combined.shape}")
        print(f"Date range: {self.df_combined['date'].min()} to {self.df_combined['date'].max()}")
        return self.df_combined
    
    def data_quality_check(self):
        """
        Validate data quality and report issues.
        """
        print("\nDATA QUALITY REPORT")
        print("="*60)
        
        # Missing values
        print(f"Missing Close prices: {self.df_combined['Close'].isna().sum()}")
        print(f"Missing news: {self.df_combined['all_headlines'].isna().sum()}")
        
        # Price statistics
        print(f"\nPrice Statistics:")
        print(f"  Min: ${self.df_combined['Close'].min():.2f}")
        print(f"  Max: ${self.df_combined['Close'].max():.2f}")
        print(f"  Mean: ${self.df_combined['Close'].mean():.2f}")
        print(f"  Volatility: {self.df_combined['Close'].pct_change().std():.4f}")
        
        # News statistics
        print(f"\nNews Statistics:")
        print(f"  Total headlines: {len(self.df_combined)}")
        print(f"  Days covered: {self.df_combined['date'].nunique()}")
        print(f"  Avg words per day: {self.df_combined['all_headlines'].str.split().str.len().mean():.0f}")
```

**Acceptance Criteria:**
- Stock prices download successfully
- News data loads and filters by ticker
- Data aligned by date with inner join
- No missing values in key columns (Close price, headlines)
- Data quality report generated
- Date range spans 6+ months

**Expected Output:**
```
REAL DATA COLLECTION RESULTS
═════════════════════════════════════════════════════════════

Downloaded AAPL prices from 2024-01-01 to 2024-06-30...
Downloaded 127 days of price data

Loaded news data from financial_news.csv...
Loaded 234 news articles for AAPL

Combined dataset shape: (127, 7)
Date range: 2024-01-02 to 2024-06-28

DATA QUALITY REPORT
═════════════════════════════════════════════════════════════
Missing Close prices: 0
Missing news: 0

Price Statistics:
  Min: $170.50
  Max: $210.75
  Mean: $192.30
  Volatility: 0.0145 (1.45% daily)

News Statistics:
  Total headlines: 127
  Days covered: 127
  Avg words per day: 45

✓ Data quality: EXCELLENT
```

---

### Notebook 2: task_2_sentiment_scoring.ipynb

```python
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class RealDataSentimentAnalyzer:
    def __init__(self, tfidf_vectorizer, lr_model, finbert_model_path="ProsusAI/finbert"):
        """
        Initialize all 3 sentiment analyzers.
        
        Parameters:
        - tfidf_vectorizer: From Week 2
        - lr_model: Logistic Regression from Week 2
        - finbert_model_path: FinBERT path
        """
        # VADER (Week 1)
        self.vader = SentimentIntensityAnalyzer()
        
        # Logistic Regression (Week 2)
        self.vectorizer = tfidf_vectorizer
        self.lr_model = lr_model
        
        # FinBERT (Week 3)
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(finbert_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def score_headline_vader(self, headline):
        """Score using VADER."""
        scores = self.vader.polarity_scores(headline)
        # Return compound score (-1 to +1)
        return scores['compound']
    
    def score_headline_lr(self, headline):
        """Score using Logistic Regression."""
        features = self.vectorizer.transform([headline])
        proba = self.lr_model.predict_proba(features)[0]
        # Return probability difference: positive - negative
        return proba[2] - proba[0]
    
    def score_headline_finbert(self, headline):
        """Score using FinBERT."""
        inputs = self.tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        # Return probability difference: positive - negative
        return (probs[2] - probs[0]).item()
    
    def score_all_headlines(self, headlines):
        """
        Score all headlines with all 3 methods.
        Returns: DataFrame with sentiments
        """
        results = []
        
        for i, headline in enumerate(headlines):
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(headlines)} headlines...")
            
            vader_score = self.score_headline_vader(headline)
            lr_score = self.score_headline_lr(headline)
            finbert_score = self.score_headline_finbert(headline)
            
            results.append({
                'headline': headline,
                'vader_sentiment': vader_score,
                'lr_sentiment': lr_score,
                'finbert_sentiment': finbert_score,
                'mean_sentiment': (vader_score + lr_score + finbert_score) / 3
            })
        
        df_sentiments = pd.DataFrame(results)
        return df_sentiments
```

**Acceptance Criteria:**
- Sentiment scores calculated for all headlines
- Scores in reasonable ranges (VADER -1 to +1, others -1 to +1)
- Processing time tracked (should be <1hr for 100+ headlines)
- Model agreement analysis (correlation between methods)

**Expected Output:**
```
SENTIMENT SCORING RESULTS
═════════════════════════════════════════════════════════════

Processed 127 headlines with 3 sentiment methods...

SAMPLE SENTIMENTS:
Headline: "Apple beats earnings expectations, raises guidance"
  VADER: 0.72
  Logistic Regression: 0.68
  FinBERT: 0.81
  Mean: 0.74

Headline: "Tech stocks decline amid rising interest rates"
  VADER: -0.58
  Logistic Regression: -0.64
  FinBERT: -0.71
  Mean: -0.64

Headline: "Market shows mixed signals for growth stocks"
  VADER: 0.12
  Logistic Regression: 0.08
  FinBERT: 0.15
  Mean: 0.12

MODEL AGREEMENT ANALYSIS
═════════════════════════════════════════════════════════════
Correlation Matrix:
                    VADER   Logistic Reg   FinBERT
VADER               1.00    0.78          0.82
Logistic Reg        0.78    1.00          0.85
FinBERT             0.82    0.85          1.00

Insight:
- All models highly correlated (r > 0.78)
- FinBERT slightly more confident (higher absolute scores)
- ML and FinBERT most similar (r=0.85)
```

---

### Notebook 3: task_3_correlation_analysis.ipynb

```python
import matplotlib.pyplot as plt
from scipy import stats

class CorrelationAnalysis:
    def __init__(self, df_combined):
        """
        Initialize correlation analysis.
        
        df_combined: DataFrame with dates, prices, and sentiments
        """
        self.df = df_combined.copy()
        self._calculate_returns()
    
    def _calculate_returns(self):
        """Calculate daily and next-day returns."""
        self.df['daily_return'] = self.df['Close'].pct_change()
        self.df['next_day_return'] = self.df['daily_return'].shift(-1)
    
    def calculate_correlations(self):
        """
        Calculate correlation between each sentiment method
        and next-day returns.
        
        Returns: DataFrame with correlations and p-values
        """
        correlations = []
        
        for method in ['vader_sentiment', 'lr_sentiment', 'finbert_sentiment']:
            # Remove NaN values
            valid_idx = ~(self.df[method].isna() | self.df['next_day_return'].isna())
            x = self.df[valid_idx][method]
            y = self.df[valid_idx]['next_day_return']
            
            # Calculate Pearson correlation
            r, p_value = stats.pearsonr(x, y)
            
            correlations.append({
                'method': method.replace('_sentiment', '').upper(),
                'correlation': r,
                'p_value': p_value,
                'n_samples': valid_idx.sum()
            })
        
        df_corr = pd.DataFrame(correlations)
        return df_corr
    
    def generate_trading_signals(self, threshold=0.3):
        """
        Generate buy/sell/hold signals based on sentiment.
        
        Parameters:
        - threshold: Sentiment threshold for signals
        
        Returns: DataFrame with signals
        """
        def signal_from_sentiment(sentiment):
            if pd.isna(sentiment):
                return 0
            elif sentiment > threshold:
                return 1  # Buy
            elif sentiment < -threshold:
                return -1  # Sell
            else:
                return 0  # Hold
        
        self.df['signal_vader'] = self.df['vader_sentiment'].apply(signal_from_sentiment)
        self.df['signal_lr'] = self.df['lr_sentiment'].apply(signal_from_sentiment)
        self.df['signal_finbert'] = self.df['finbert_sentiment'].apply(signal_from_sentiment)
        
        return self.df[['date', 'signal_vader', 'signal_lr', 'signal_finbert']]
    
    def visualize_sentiment_vs_price(self):
        """
        Create time series plot showing sentiment and price movement.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: VADER sentiment vs returns
        axes[0].plot(self.df['date'], self.df['vader_sentiment'], label='VADER Sentiment', color='blue', alpha=0.7)
        axes[0].scatter(self.df['date'], self.df['next_day_return'] * 100, label='Next-day Return (%)', color='red', alpha=0.3, s=20)
        axes[0].set_title('VADER Sentiment vs Stock Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Logistic Regression sentiment vs returns
        axes[1].plot(self.df['date'], self.df['lr_sentiment'], label='LR Sentiment', color='green', alpha=0.7)
        axes[1].scatter(self.df['date'], self.df['next_day_return'] * 100, label='Next-day Return (%)', color='red', alpha=0.3, s=20)
        axes[1].set_title('Logistic Regression Sentiment vs Stock Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: FinBERT sentiment vs returns
        axes[2].plot(self.df['date'], self.df['finbert_sentiment'], label='FinBERT Sentiment', color='purple', alpha=0.7)
        axes[2].scatter(self.df['date'], self.df['next_day_return'] * 100, label='Next-day Return (%)', color='red', alpha=0.3, s=20)
        axes[2].set_title('FinBERT Sentiment vs Stock Returns')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

**Acceptance Criteria:**
- P-values computed and significance assessed
- Trading signals generated (buy/sell/hold)
- Time series plots created (sentiment vs returns)
- Statistical summary table generated

**Expected Output:**
```
CORRELATION ANALYSIS RESULTS
═════════════════════════════════════════════════════════════

SENTIMENT-PRICE CORRELATION TABLE
Method                  Correlation  P-value   Significant?
─────────────────────────────────────────────────────────────
VADER                   0.32         0.0012    Yes (p<0.05)
Logistic Regression     0.38         0.0001    Yes (p<0.05)
FinBERT                 0.42         <0.0001   Yes (p<0.05)

KEY FINDINGS:
- All methods significantly predict next-day returns
- FinBERT has strongest correlation (0.42)
- ML method outperforms VADER (0.38 vs 0.32)
- Ranking consistent with theory: FinBERT > ML > VADER

TRADING SIGNALS GENERATED
Method                  Buy Signals  Sell Signals  Hold Days
─────────────────────────────────────────────────────────────
VADER                   28          18           81
Logistic Regression     32          25           70
FinBERT                 35          22           70

INTERPRETATION:
- FinBERT most bullish (35 buy signals)
- Threshold=0.3 seems reasonable
- Ready for backtesting in Week 5
```

---

## Setup for Google Colab

```python
# Cell 1: Install packages
!pip install yfinance pandas numpy scikit-learn transformers torch scipy

# Cell 2: Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Cell 3: Download example data
ticker = "AAPL"
df_prices = yf.download(ticker, start="2024-01-01", end="2024-06-30", progress=False)
print(f"✓ Downloaded {len(df_prices)} days of {ticker} prices")

# Cell 4: Load sentiment models from Week 1-3
# (Instructions to load saved models)

print("✓ Setup complete! Ready for Week 4")
```
---

## Key Insights for Week 4

### Why Real Data Often Disappoints

Lab results (Financial PhraseBank): 0.87 F1 for FinBERT

Real data correlation: r=0.42 for FinBERT

This is NOT a failure. Here's why:

1. **Lab data is clean & curated:** Financial PhraseBank has clear sentiment labels
2. **Real data is noisy:** Headlines from many sources, ambiguous sentiment
3. **Time lag issues:** When sentiment published vs when price moves?
4. **Other factors matter:** Earnings, economic data, market-wide sentiment
5. **Correlation ≠ accuracy:** Even r=0.42 is useful for trading (Week 5)

### Expected Outcome

```
THEORY (Lab) vs PRACTICE (Real Data)
═════════════════════════════════════════════════════════════

                    Lab F1      Real Correlation
VADER               0.75        0.30-0.35
Logistic Regression 0.80        0.35-0.40
FinBERT             0.87        0.40-0.45

Key Pattern:
- Rankings consistent (FinBERT > ML > VADER)
- Magnitudes smaller (0.87 F1 → 0.42 correlation)
- Still statistically significant (p < 0.001)
- Sufficient for trading signals (Week 5)
```

### Why Correlation Matters More Than Accuracy Here

In Week 1-3, we measured accuracy on labeled data.

In Week 4, we measure correlation with unlabeled data (price).

Why? Because:
1. Price is the "ground truth" we actually care about
2. Accuracy on Financial PhraseBank is indirect
3. Weak correlation (r=0.4) can still be profitable if consistent

Example:
- 0.87 F1 accuracy but r=0.42 with price → useful
- 0.80 F1 accuracy but r=0.38 with price → still useful (smaller gap)
- VADER can beat ML in trading despite lower accuracy (depends on execution)

Week 5 backtesting will answer: which actually makes money?

---
