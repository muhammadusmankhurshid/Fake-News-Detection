# Fake News Detection System

## Overview
This project implements a machine learning system for detecting fake news articles using natural language processing and multiple classification algorithms. The system employs K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machines (SVM) to classify news articles as either genuine or fake based on their textual content.

## Problem Statement
The proliferation of fake news in digital media has become a significant concern. This system aims to automatically identify potentially false news articles by analyzing text patterns, writing styles, and linguistic features that are commonly associated with misinformation.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- re (regular expressions)

## Installation
Install the required packages using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
```

## Data Requirements
The project uses two CSV files with semicolon (;) as the delimiter:
- `train.csv`: Training dataset containing news articles with labels (fake/genuine)
- `test.csv`: Test dataset for model evaluation

Each dataset should contain:
- `text`: The news article content
- `label`: Binary classification (fake/genuine)

## Text Processing Pipeline
1. **Preprocessing**
   - Lowercase conversion
   - URL removal
   - Special character removal
   - Tokenization
   - Stopword removal
   - Word stemming

2. **Feature Engineering**
   - TF-IDF vectorization with bi-gram support
   - Feature selection using Chi-square test
   - Dimensionality reduction for optimal performance

## Classification Models
The system implements three different classifiers:
1. **K-Nearest Neighbors (KNN)**
   - Pattern-based classification
   - Hyperparameter tuning for optimal k-value

2. **Naive Bayes**
   - Probability-based classification
   - Efficient handling of high-dimensional text data

3. **Support Vector Machine (SVM)**
   - Linear and RBF kernel implementation
   - Optimal for text classification tasks

## Model Evaluation
The system evaluates detection accuracy using:
- Cross-validation scores
- Precision (false positive minimization)
- Recall (false negative minimization)
- F1-Score (balanced metric)
- ROC curves and AUC scores
- Confusion matrices

## Advanced Analysis Features
- Hyperparameter optimization using GridSearchCV
- Text cluster analysis using K-means
- Silhouette analysis for cluster validation
- Feature importance visualization
- Model performance comparison

## Usage Instructions
1. Data Preparation:
   ```python
   train_df = pd.read_csv('train.csv', delimiter=';')
   test_df = pd.read_csv('test.csv', delimiter=';')
   ```

2. Text Preprocessing:
   ```python
   train_df['text_cleaned'] = train_df['text'].apply(preprocess_text_advanced)
   ```

3. Model Training and Evaluation:
   ```python
   # Train models
   knn.fit(X_train, y_train)
   nb.fit(X_train, y_train)
   svm.fit(X_train, y_train)
   
   # Evaluate performance
   evaluate_with_cross_validation(model, X_train, y_train)
   ```

## Output and Visualization
The system generates:
- Model accuracy comparisons
- Feature importance rankings
- ROC curves for classification performance
- Clustering visualizations
- Detailed performance metrics tables

## Performance Metrics
The system tracks:
- Overall classification accuracy
- False positive rates (incorrectly flagged genuine news)
- False negative rates (missed fake news)
- Model confidence scores
- Classification speed and efficiency

## Future Enhancements
- Deep learning model integration
- Real-time news article analysis
- Multi-language support
- Source credibility scoring
- Explanation generation for classifications

## Notes for Deployment
- Regular model retraining recommended
- Performance monitoring needed
- Consider computational resources for scaling
- Implement rate limiting for production use
