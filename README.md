# COMP7404-Project

# Project Title

Machine Learning Techniques for Short-Term Hang Seng Futures Price Prediction

## Getting Started

This is a program to implement various machine learning algorithm based on python.

We will be using various techniques such as neural network, svm, logistic regression and random forest to implement how the machine learning ideas can be applied in trading. We will also include a file for recursive features elimination. 
*Dataset:

Hang Seng Futures minute level price data for 2014. The dataset is already cleaned with the first column as UP, DOWN or FLAT after N-minute. The rest are features which include commonly used indicators, such as ROC, RSI. The data are also standardised from 0 to 1.
Datasets are separated by training data (from 2014-01-01 to 2014-06-30) and testing data (from 2014-07-01 to 2014-12-31).

## How to use

Users should first download the dataset. First ran the RFE.py file which is the features selection program to select the top 10 features for each minute set. Next, users can run each of the machine learning program (including Neural Network, SVM, Random Forest and Logistic Regression) to compare the predicted result with the actual result. 

### Prerequisites

*Requirements:
Python 3.7.0 (https://docs.python.org/3/using/index.html)
numpy 1.15.2 (https://scipy.org/install.html)
matplotlib 3.0.0 (https://matplotlib.org/faq/installing_faq.html) 
pandas 0.23.4 (https://pandas.pydata.org/pandas-docs/stable/install.html)
sklearn 0.20.1 (https://scikit-learn.org/stable/)

Link: https://github.com/lch81592/COMP7404-Project
