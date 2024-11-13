# Lab 1: NYSE Data Analysis & Predictive Maintenance with PyTorch
This repository includes notebooks that explore financial and predictive maintenance datasets using deep learning models for regression and classification tasks with PyTorch. The primary objective is to gain hands-on experience with building DNN (Deep Neural Network) and MLP (Multi-Layer Perceptron) models for predictive analysis.
# Project Overview
## Part 1: NYSE Data Analysis and Regression Model Development
Analyze New York Stock Exchange data to predict stock price movements and explore stock fundamentals using DNN models. The two main objectives are:

Conduct exploratory data analysis (EDA) for insights.
Build regression models to predict stock price metrics.
## Part 2: Predictive Maintenance Classification
Apply classification techniques to predictive maintenance data to anticipate machinery failures. Objectives include:

Use EDA for sensor data analysis.
Develop and optimize a multi-class classification model for failure type prediction.
### Dataset Information
NYSE Data: [New York Stock Exchange Data](https://www.kaggle.com/datasets/dgawlik/nyse)
Contains historical price data and financial indicators for NYSE-listed companies.
Predictive Maintenance Data: [Predictive Maintenance Data: Kaggle - Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
Contains sensor data aimed at predicting machinery maintenance needs.
# Project Workflow
## Part 1: NYSE Data Analysis
### 1. Exploratory Data Analysis (EDA)
Visualize trends and clean data for modeling insights.
### 2. DNN for Regression
Design a regression model using DNN to predict stock prices.
Optimize using MLP architecture for accurate prediction.
### 3. Hyperparameter Tuning with GridSearchCV
Tune parameters (e.g., learning rate, optimizer, epochs) using GridSearch to improve performance.
### 4. Training Visualization
Plot and analyze training/validation Loss and Accuracy vs. Epochs graphs.
### 5. Regularization
Implement dropout and weight decay to reduce overfitting.
## Part 2: Predictive Maintenance - Classification Model


### 1. Data Preprocessing and EDA
Standardize data, perform EDA, and balance classes via data augmentation.
### 2. DNN Model for Classification
Build a DNN for predictive maintenance, targeting failure types in equipment.
### 3. Hyperparameter Tuning
Use GridSearch to find optimal settings for improved model efficiency.
###4. Training Visualization
Plot Loss and Accuracy vs. Epochs for performance assessment.
### 5. Evaluation and Regularization
Assess model using accuracy, sensitivity, F1 score, and implement regularization techniques.

# Notebooks

## prices.ipynb:
 Regression analysis on NYSE price data.
## prices-split-adjusted.ipynb:
Additional NYSE regression analysis with engineered features.
## fundamentals.ipynb:
 Regression model on NYSE financial fundamentals.
## predictive-maintenance.ipynb: 
Multi-class classification for predictive maintenance.

# Requirements
  - PyTorch
  - scikit-learn
  - Pandas
  - Matplotlib
