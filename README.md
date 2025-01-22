# Instagram Content Analysis and Engagement Prediction

## Overview
This project analyzes Instagram content and predicts post engagement (like counts) using machine learning techniques. It consists of two key components:

1. **Classification**: Categorizing Instagram accounts into predefined content categories based on user profiles and posts.
2. **Regression**: Predicting the number of likes on Instagram posts based on features derived from the posts and user metrics.

## Key Features
- Hybrid model combining DistilBERT embeddings for text features and engineered numerical features for classification.
- XGBoost regression model to predict post engagement based on structured data.
- Data preprocessing steps including missing value handling, feature scaling, and outlier detection.
- Achieved high classification accuracy and reliable predictions for post engagement metrics.

## Prerequisites
- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `scikit-learn`, `transformers`, `xgboost`, `torch`, `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/instagram-engagement-prediction.git
   cd instagram-engagement-prediction

