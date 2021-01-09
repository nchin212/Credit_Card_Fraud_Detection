# Credit_Card_Fraud_Detection

## Overview

- Analyzed credit card transactions made by European cardholders by classifying them as safe or fraudulent
- Tried 2 different oversampling techniques, RandomOverSampler and Synthetic Minority Oversampling Technique (SMOTE)
- Implemented logistic regression, random forest and adaboost on imbalanced data and oversampled data
- Introduced various metrics to evaluate classification models such as confusion matrix, classification report and Area Under the Curve (AUC)
- Random Forest with RandomOverSampler achieved the highest f1 score of 87.7%

## Tools Used

- Language: Python 
- Packages: pandas, numpy, matplotlib, seaborn, sklearn, imblearn
- Data: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Topics: Python, Classification, Data Imbalance, Oversampling, RandomOverSampler, SMOTE, Logistic Regression, Random Forest, Adaboost

## Data

The dataset is taken from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by credit cards in September 2013 by European cardholders. Due to confidentiality issues, Kaggle is unable to provide the original features and more background information about the data. It contains the following columns:

| Variable         | Description                                                                       |
|:-----------------|:----------------------------------------------------------------------------------|
| V1, V2, ..., V28 | Principle components obtained with PCA                                            |
| Amount           | Transaction amount                                                                |
| Time             | Seconds elapsed between each transaction and the first transaction in the dataset |
| Class            | Response variable, takes value 1 in case of fraud and 0 otherwise                 |

## Exploratory Data Analysis

Histograms of Variables   |  
:-------------------------:|
![alt text](https://github.com/nchin212/Credit_Card_Fraud_Detection/blob/gh-pages/plots/hist1.png) |  

Transaction Amount over Time  |  
:-------------------------:|
![alt text](https://github.com/nchin212/Credit_Card_Fraud_Detection/blob/gh-pages/plots/line1.png) |  

## Data Cleaning

- Normalized the `Amount` and `Time` columns

## Data Imbalance

When we check the `Class` column, there is severe class imbalance.

![alt text](https://github.com/nchin212/Credit_Card_Fraud_Detection/blob/gh-pages/plots/bar1.png) 

One of the ways to resolve data imbalance is to resample the dataset. This can be done by undersampling or oversampling. In this case, we will use oversampling as undersampling would result in too little data.

## Oversampling

**RandomOverSampler -** It involves oversampling the minority class by picking samples at random with replacement

**SMOTE -** It first selects a minority class instance 'a' at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors 'b' at random and connecting 'a' and 'b' to form a line segment. The synthetic instances are generated as a convex combination of the two chosen instances 'a' and 'b'.

## Model Building

The following models were chosen to classify the transactions as safe or fraudulent:

- Logistic Regression
- Random Forest (Bagging)
- Adaboost (Boosting)

## Results

|                          | Accuracy | Precision |   Recall | F1-score |      AUC |
|-------------------------:|---------:|----------:|---------:|---------:|---------:|
|        LogReg Imbalanced | 0.999274 |  0.877551 | 0.632353 | 0.735043 | 0.816106 |
| LogReg RandomOverSampler | 0.974580 |  0.055070 | 0.926471 | 0.103960 | 0.950563 |
|             LogReg SMOTE | 0.973245 |  0.052829 | 0.933824 | 0.100000 | 0.953566 |
|            RF Imbalanced | 0.999579 |  0.923729 | 0.801471 | 0.858268 | 0.900683 |
|     RF RandomOverSampler | 0.999637 |  0.948718 | 0.816176 | 0.877470 | 0.908053 |
|                 RF SMOTE | 0.999520 |  0.836879 | 0.867647 | 0.851986 | 0.933689 |
|           Ada Imbalanced | 0.999485 |  0.870968 | 0.794118 | 0.830769 | 0.896965 |
|    Ada RandomOverSampler | 0.989748 |  0.127016 | 0.926471 | 0.223404 | 0.958160 |
|                Ada SMOTE | 0.983428 |  0.083333 | 0.941176 | 0.153110 | 0.962336 |

Since this is an unbalanced dataset, accuracy is not the correct metric to use to compare the models as the models will tend to predict the majority class, resulting in high accuracy rates even though the minority class has been classified wrongly. Instead, we should compare the models using their f1-scores.

![alt text](https://github.com/nchin212/Credit_Card_Fraud_Detection/blob/gh-pages/plots/bar2.png) 

## Relevant Links

**Jupyter Notebook :** https://nchin212.github.io/Credit_Card_Fraud_Detection/credit.html

**Portfolio :** https://nchin212.github.io/post/credit_fraud/
