NYSE Data Analysis and Predictive Maintenance with Deep Learning (PyTorch)
This repository contains Jupyter notebooks for two key projects focused on financial and predictive maintenance data. Both projects leverage deep learning models built with PyTorch for classification and regression tasks.

Project 1: NYSE Data Analysis and Regression Modeling
The objective of this project is to develop skills in regression tasks by building Deep Neural Network (DNN) models with PyTorch, specifically targeting financial data from the NYSE. Through exploratory data analysis and model optimization, this project aims to predict stock-related metrics and identify patterns in stock price data.

Dataset
Source: New York Stock Exchange Data on Kaggle
Description: Historical stock prices and fundamental indicators for NYSE-listed companies.
Workflow
Exploratory Data Analysis (EDA)

Conducted EDA to clean, visualize, and understand the dataset, identifying significant trends and patterns.
DNN Model for Regression

Designed a Multi-Layer Perceptron (MLP) in PyTorch to perform regression on stock prices.
Optimized model architecture to capture stock price complexities.
Hyperparameter Tuning

Used GridSearchCV from scikit-learn to find optimal parameters (learning rate, optimizer choice, epochs, etc.).
Fine-tuned parameters to enhance model efficiency and accuracy.
Training Visualization

Plotted Loss and Accuracy vs. Epochs for both training and testing datasets.
Analyzed convergence and highlighted potential improvement areas.
Regularization Techniques

Applied dropout and weight decay to reduce overfitting and improve generalization.
Compared results with and without regularization to assess performance impact.
Notebooks
nyse-prices.ipynb: EDA and regression model development for NYSE price data.
nyse-fundamentals.ipynb: EDA and regression model development based on company financial indicators.
Project 2: Predictive Maintenance with Multi-Class Classification
The goal of this project is to develop proficiency in PyTorch by building a DNN model for a multi-class classification task focused on predictive maintenance. The task is to predict machine failure types based on sensor data, optimizing the model for improved prediction accuracy.

Dataset
Source: Predictive Maintenance Dataset on Kaggle
Description: Machine sensor data aimed at predicting maintenance needs by classifying different failure types.
Workflow
Data Preprocessing

Cleaned and standardized the dataset for better model performance.
Exploratory Data Analysis (EDA)

Visualized data to identify feature relationships and insights for modeling.
Data Augmentation

Applied data augmentation to balance the dataset and enhance model generalization.
Model Architecture

Built a deep neural network for multi-class classification, targeting failure type prediction.
Hyperparameter Tuning

Used GridSearch to optimize parameters like learning rate, optimizer type, epochs, and model structure.
Training Visualization

Analyzed Loss vs. Epochs and Accuracy vs. Epochs plots to understand model behavior and performance.
Model Evaluation

Evaluated the model using metrics such as accuracy, sensitivity, and F1 score for training and testing datasets.
Regularization Techniques

Tested various regularization methods and compared outcomes to improve model robustness.
Notebook
predictive-maintenance.ipynb: Comprehensive EDA, model training, evaluation, and optimization for predictive maintenance tasks.
Requirements
To run these projects, install the following libraries:

Python 3.x
PyTorch
Scikit-Learn
Pandas
Matplotlib
Seaborn