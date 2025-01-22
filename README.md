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
Project Structure
data/: Contains dataset and preprocessing scripts.
models/: Includes trained models and model-building scripts for both classification and regression tasks.
notebooks/: Jupyter notebooks for analysis and model evaluation.
outputs/: Contains predictions and evaluation metrics.
requirements.txt: List of dependencies for the project.
How It Works
Classification
Data preprocessing steps like balancing the dataset, handling missing values, and combining captions and biographies into a single feature.
Text embeddings extracted using DistilBERT to represent user bios and captions.
Numerical features such as follower count, post count, etc., normalized and passed through a dense layer.
A hybrid neural network architecture combines text and numerical features for classification.
Model trained using AdamW optimizer with a learning rate of 5e-5 and CrossEntropyLoss.
Regression
Data cleaning steps such as imputing missing values and removing outliers using Isolation Forest.
Log transformation applied to the target variable to normalize its distribution.
XGBoost model configured for regression with hyperparameters optimized for structured data.
Model trained to predict the like count of Instagram posts.
Evaluation metrics such as MSE, MAE, and R-squared are calculated for model performance.
Results
Classification: High accuracy in categorizing Instagram accounts into predefined content categories.
Regression: Reliable predictions of like counts with minimal error.
Future Improvements
Expand the dataset for better generalization.
Experiment with transformer-based architectures for regression tasks.
Incorporate advanced feature selection techniques for enhanced model performance.
