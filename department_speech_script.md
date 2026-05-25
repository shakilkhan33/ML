# Department Speech Script — Stock Price Prediction Using Machine Learning
**Speaker:** Shakil Khan | CUNY
**Duration:** ~20 minutes | ~2,800 words at 140 wpm
**Audience:** Department faculty, students, and staff

---

> **SPEAKER NOTES:** Sections marked [PAUSE] = take a breath, let the point land.
> Sections marked [SLIDE] = advance your slide.
> Estimated timing is shown in brackets at each section heading.

---

## OPENING — The Hook [0:00–1:30]

Good [morning / afternoon], everyone.

I want to start with a simple question.

If I told you that I built a computer program — a machine learning system — that could look at today's stock price for Apple and tell you what it will be tomorrow, with an average error of less than two dollars... would you believe me?

[PAUSE]

Maybe you'd be skeptical. Maybe you'd wonder: if this actually worked reliably, wouldn't everyone be rich by now?

That skepticism is exactly the right instinct — and it's precisely the question this project is designed to explore.

My name is Shakil Khan, and today I'm presenting my Machine Learning final project: **"Stock Price Prediction — Comparing Six Machine Learning Models on Apple Stock."**

Over the next twenty minutes, I'm going to walk you through how I framed the problem, how I built the system, what each model does, what the results showed, and — most importantly — what those results actually mean.

[SLIDE]

---

## WHY THIS PROBLEM? [1:30–3:30]

Let's talk about why stock price prediction is one of the most studied — and most difficult — problems in machine learning.

Stock markets move the global economy. Apple, the company at the center of this project, has a market capitalization of over three trillion dollars. Every day, millions of traders make decisions based on price forecasts. Even a small improvement in prediction accuracy can translate into enormous financial consequences.

But here's what makes this problem so hard:

Markets are what economists call *nearly efficient*. That means, in theory, all publicly available information is already priced in. The price you see right now already reflects everything the market collectively knows. Tomorrow's price is determined by *new* information — and new information, by definition, is unpredictable.

So why use machine learning at all?

Because while perfect prediction is impossible, *pattern recognition* is not. Prices don't move in a completely random way. There are trends, momentum signals, and volatility patterns that models can learn from.

The goal of this project is not to beat the market. The goal is to rigorously test whether machine learning models can outperform the simplest possible baseline — and to understand *why* they succeed or fail.

[PAUSE]

That baseline is called the **Naive Random Walk** — the assumption that tomorrow's price equals today's price. It sounds too simple to be useful, but as you'll see, it's actually a very strong benchmark.

[SLIDE]

---

## THE DATA [3:30–6:00]

My dataset covers five S&P 500 stocks: Apple, Tesla, Amazon, Visa, and Microsoft. I pulled closing prices from January 2017 through December 2019 — approximately 754 trading days per stock — using Yahoo Finance through a Python library called `yfinance`.

Why those three years? Because 2017 through 2019 represents a stable bull market period with no pandemic distortions or extreme black-swan events. It gives us a clean, representative window for evaluating model performance.

[SLIDE]

Before any modeling begins, the data goes through a careful cleaning pipeline.

I parse dates, remove duplicates, sort chronologically, and drop any row with missing prices. This matters because machine learning models don't tolerate dirty data — a single missing value or out-of-order date can corrupt an entire training run.

The key target variable is simple: **tomorrow's closing price for Apple**. Everything the model knows about today is used to predict what happens tomorrow.

[SLIDE]

### 3a. Log Returns

Before I engineer features, I convert the price series into log returns.

That step matters because raw stock prices are hard to compare directly across companies. Apple and Tesla may trade at very different price levels, but a log return puts their daily movement on the same scale. It also makes the changes easier to interpret statistically, because returns are more stable and closer to a normal-shaped distribution than raw prices.

On this slide, the table shows the day-to-day log returns for Apple, Tesla, Amazon, Visa, and Microsoft. A positive value means the stock moved up that day, and a negative value means it moved down. You can also see how small most daily changes are, which is exactly why returns are a better input for exploratory analysis than raw prices.

In simple terms, log returns let me focus on movement instead of absolute price level. That gives the model and the analysis a cleaner, more comparable view of market behavior.

[SLIDE]

### Feature Engineering

This is where the interesting work begins.

Raw price data alone is not very informative. What tells a better story are the *derived signals* — the features we engineer from the price series.

I built 11 input features across three categories:

**Momentum features** — the percentage price change over the past 1, 2, 3, 4, and 5 days. These capture short-term directional trends.

**Trend features** — the 5-day and 10-day simple moving averages, and their ratio. When the short-term average is above the long-term average, that's a classic bullish signal.

**Risk features** — rolling 5-day volatility, which measures how wildly the price has been swinging recently. And a momentum indicator that compares today's price to the 10-day average.

These 11 features, together with today's raw price, form the input to every tabular model.

[PAUSE]

One critical rule I followed: the StandardScaler — the tool that normalizes all features to the same scale — is **fit only on the training data**. Never on the full dataset. If you scale using statistics from the test set, you are leaking future information into your model. That's a subtle but fatal mistake that invalidates your results.

[SLIDE]

---

## THE MODELS [6:00–10:00]

I trained and evaluated six models, divided into three categories.

[SLIDE]

### Category 1: Linear — Linear Regression

Linear Regression is the simplest possible model. It fits a straight-line relationship between the 11 input features and tomorrow's price, by minimizing the sum of squared prediction errors.

You might wonder: why use such a basic model for something as complex as stock prices?

The answer is that stock prices are highly *autocorrelated* — today's price is the single strongest predictor of tomorrow's price. Linear Regression captures this relationship efficiently, without the risk of overfitting. It often punches far above its weight class.

[SLIDE]

### Category 2: Tree-Based Ensembles — Random Forest, Decision Tree, XGBoost

**Decision Tree** splits the data recursively — at each step, it finds the feature and threshold that best reduces prediction error. I constrained it to a maximum depth of 5 layers and required each leaf to contain at least 5 training samples. Without these constraints, a decision tree will memorize the training data perfectly and fail completely on new data — a phenomenon called overfitting.

**Random Forest** builds 50 of these trees simultaneously, each trained on a random subset of the data. The final prediction is the average across all 50 trees. This averaging process dramatically reduces variance — the model becomes more stable and generalizes better.

**XGBoost** takes a fundamentally different approach. Instead of building trees independently and averaging, it builds them *sequentially*. Each new tree is specifically trained to correct the errors made by all the previous trees. This process — called gradient boosting — often achieves the best accuracy on tabular data.

[PAUSE]

I want to be precise about why I chose these three specific ensemble methods. They represent three distinct philosophies: single-tree simplicity, parallel averaging, and sequential error correction. Comparing them teaches us something fundamental about how learning from data actually works.

[SLIDE]

### Category 3: Deep Learning — Simple RNN and LSTM

The deep learning models require a completely different data format. Instead of 11 features for a single day, they receive a **sequence** — the last 30 closing prices in order — as input.

**Simple RNN** processes this sequence one day at a time, maintaining a hidden state that is updated at each step. Think of it as a short-term memory — it can remember what happened over the last few steps, but struggles to learn patterns from more than 10 to 20 days back. This is known as the vanishing gradient problem.

**LSTM** — Long Short-Term Memory — solves this with three gating mechanisms: a forget gate that decides what to discard, an input gate that decides what to store, and an output gate that decides what to release. This allows LSTM to selectively remember important patterns across the full 30-day window, not just the last few days.

Both models were trained with early stopping — if validation loss stopped improving for five consecutive epochs, training was automatically halted. This prevents overfitting without manual tuning.

[SLIDE]

---

## RESULTS [10:00–14:00]

Now let's talk about what actually happened when we tested these models on 20% of the data they had never seen before.

[PAUSE]

The evaluation metric I use is **Mean Absolute Error** — MAE — measured in US dollars. This tells you the average dollar amount by which a prediction was wrong. Lower is better.

I also report **R-squared** — a measure of how much of the variance in the actual prices the model explains. A perfect model gets 1.0. A model that just predicts the average every time gets 0.0. A negative R-squared means the model is literally worse than predicting the average.

[SLIDE]

### The Naive Baseline

Before any ML model, the naive baseline — predict that tomorrow equals today — achieves a MAE of roughly 1 to 2 dollars on Apple stock in this period. That's the floor. Every model must beat this, or it has added no value.

[SLIDE]

### Tabular Model Results

Linear Regression matched or slightly beat the naive baseline — confirming the intuition that price autocorrelation is real and models that capture it efficiently tend to perform well.

Random Forest and XGBoost both showed meaningful improvements over the baseline. Their feature importance analysis revealed something revealing: **today's raw price is overwhelmingly the most important feature** in both models — often accounting for 80 to 90 percent of the importance score. The engineered features like moving averages and volatility contribute, but they are secondary.

This tells us something important about what the models actually learned. They are, to a large degree, learning a sophisticated version of the naive baseline — "tomorrow will be close to today" — but with enough nuance from the technical features to reduce the error slightly.

Decision Tree performed worse than the ensemble methods, as expected. Single trees, even when depth-limited, are more sensitive to noise than averaged or boosted ensembles.

[SLIDE]

### Sequential Model Results

The RNN and LSTM results are where things get interesting.

On the 30-day sliding window test set, both models tracked the actual price curve remarkably well visually. The predictions follow the general trend and even capture some of the directional turns.

However, there is an important caveat. RNN and LSTM use a 30-day lookback window, which means their test set starts 30 days later than the tabular model test set. The MAE values are comparable in scale — we are talking US dollars, same stock — but they are not perfectly equivalent comparisons.

[PAUSE]

If you look at the prediction curves carefully, you will notice something that is both expected and slightly sobering: every model, including LSTM, shows a prediction that is essentially a smoothed, lagged version of the actual price. The model predicts approximately where the price *was*, not exactly where it *will be*. 

This is the hallmark of autocorrelation-driven prediction, and it is honest. The models are not magical. They are finding and exploiting the most stable signal available.

[SLIDE]

---

## KEY INSIGHTS [14:00–17:00]

Let me pull back from the technical details and share the three insights that I believe are most important from this project.

[SLIDE]

### Insight 1: The Naive Baseline is Powerful

This cannot be overstated. The claim "tomorrow's price equals today's price" is correct the majority of the time for large-cap stocks. Any model that fails to beat it should be discarded.

The lesson for machine learning in general: always define your baseline *before* you build your model. A 95% accuracy rate sounds impressive until you discover that always predicting "no" would give you 96% accuracy because the dataset is imbalanced.

[SLIDE]

### Insight 2: Feature Importance Reveals What Models Actually Learn

When Random Forest and XGBoost both report that today's raw price is their most important feature by a factor of 5 to 10 over any other feature, that is a diagnostic finding. It tells us that the engineered features are adding marginal value at best.

This is not a failure of engineering. It reflects a fundamental truth about stock prices: the strongest predictor of tomorrow is today. Engineering more features only helps if those features carry information that raw price does not.

[SLIDE]

### Insight 3: Time-Series Cross-Validation Matters More Than a Single Split

A single 80/20 train-test split can be misleading. Maybe the test period happened to be unusually stable, making the model look better than it is. Maybe it was unusually volatile, making every model look worse.

Time-series cross-validation using walk-forward folds gives us a much more honest picture. A model with consistent MAE across multiple test windows is genuinely robust. A model that looks great on one split but varies wildly across folds is overfitting to the specific market conditions of that split.

[SLIDE]

---

## LIMITATIONS AND FUTURE WORK [17:00–19:00]

I want to be candid about the limitations of this work, because intellectual honesty is what distinguishes good science from overselling.

[SLIDE]

**Limitation 1: No external signals.**
This model uses only price-derived features. It does not incorporate earnings announcements, macroeconomic data, interest rate changes, news sentiment, or social media signals. Real trading systems layer all of these on top of technical price features.

**Limitation 2: Transaction costs are not modeled.**
Even a model with low MAE might generate unprofitable trades in practice once you account for bid-ask spreads, brokerage fees, and the market impact of your own orders. Prediction accuracy and trading profitability are not the same thing.

**Limitation 3: The 2017–2019 period is atypically stable.**
These three years were a strong, relatively low-volatility bull market. Models trained in this period may behave very differently during a correction, a crisis, or a sudden sector rotation. Robustness to regime change is a major open question.

[PAUSE]

**Future directions:**

There are several natural extensions I would pursue with more time. First, adding sentiment features from news APIs or Twitter data to see if external signals add predictive power beyond price alone. Second, training on a longer time horizon that includes 2008, 2020, and 2022 — periods with very different market dynamics. Third, framing the problem as a classification task — predict *direction* (up or down) rather than exact price — which may be more useful and more honest given prediction uncertainty.

[SLIDE]

---

## CONCLUSION [19:00–20:00]

Let me close with where we started.

Can a machine learning model tell you what Apple's stock price will be tomorrow?

The answer is: approximately yes — with a mean absolute error of around one to two dollars. But what that model is doing is essentially learning a sophisticated version of the most obvious possible strategy: the price tomorrow will be close to the price today.

[PAUSE]

The real value of this project is not the prediction. It is the discipline.

Cleaning the data before modeling. Scaling features correctly, without leakage. Defining a baseline before claiming success. Using cross-validation to verify robustness. Analyzing residuals to detect bias. And being honest about what the feature importances actually tell us.

These practices do not just apply to stock prices. They apply to every machine learning problem you will ever work on — in finance, in healthcare, in natural language, in computer vision.

Good machine learning is not about building the most complex model. It is about asking honest questions, measuring what matters, and knowing the difference between a model that works and a model that appears to work.

Thank you for your time and attention. I'm happy to take questions on any aspect of the methodology, the results, or the code.

[PAUSE — smile, make eye contact, open the floor]

---

## Q&A PREP — Anticipated Questions

**Q: "Why didn't you use more recent data, like 2020–2024?"**
A: The 2017–2019 window was deliberately chosen for stability. Including COVID and the 2022 bear market would add important robustness testing, and that is a planned extension. The current results establish a clean baseline in a non-crisis environment.

**Q: "How does LSTM compare to the tabular models?"**
A: The comparison is not perfectly apples-to-apples due to the different test set boundaries from the 30-day lookback, but both operate in a similar MAE range. LSTM's advantage is that it can, in theory, learn long-range dependencies in the sequence — but in practice, for a single stock with strong autocorrelation, the tabular models are competitive.

**Q: "Could this be used for real trading?"**
A: Not without significant additional work. The model does not account for transaction costs, execution risk, or regime changes. Prediction accuracy and trading profitability are distinct problems.

**Q: "Why did you choose Apple specifically?"**
A: Apple was chosen as the primary modeling target because it is one of the most liquid and heavily studied stocks in the world, with a long and well-documented price history. It provides a strong, well-understood benchmark.

**Q: "What was the most challenging part?"**
A: Avoiding data leakage. The most subtle and consequential bug in time-series machine learning is fitting your scaler or feature statistics on the full dataset instead of the training set alone. I caught this early and made it a core design principle throughout the pipeline.

---

*End of script.*
