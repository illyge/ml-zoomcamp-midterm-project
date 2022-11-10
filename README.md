# Twitter Disaster Detection

## Problem statement
The goal of this project is to create a classifier, which, given a twit with some medatata, would classify it if it tells about a disaster or not.

The dataset is taken from an open Kaggle competition:
https://www.kaggle.com/competitions/nlp-getting-started

It's added to the project as `train.csv` file

## Main techniques used in the project

#### The problem belongs to the NLP domain, so the data we analyze here is purely textual. Here are some techniques that are used in the project:
1. `CountVectorizer`. This feature extraction works similarly to the `DictVectorizer`, but instead of vectorizing a category column it is applied to a column of texts. Each text is tokenized into separate words and for each word a feature is created with frequency of this word in a respective text
2. `PolynomialFeatures`. Since some words may influence the meaning of text only in combination with other words, the linear classifiers may loose some info. Creating polynomial features allows to find connections between word combinations and target
3. `SelectKBest`. After applying the previous two techniques we may end up with enormouse numbers of features, e.g. tens of thousands. This technique allows to select the most related to the target
4. `TruncatedSVD`. The final dimensionality reduction technique to reduce the number if features to hundreds

#### What and how was measured

1. 5 different models were tested:
   2. Ridge Classifier
   3. Logistic Regression
   4. Decision Tree
   5. Random Forest
   6. XGBoost
7. `F1 score`. This metric was used to evaluate the performance of the models

## How to run the project

- install requirements:
  - ```pip3 install -r requirements.txt```
- run notebook: 
  - `jupyter-lab`
- train and save model: 
  - `python train.py`
- run service locally:
  - `bentoml serve service.py:svc`

## Some results and conclusions

- The best performance was shown by Logistic Regression and Ridge Classifier. The F1 score on the test data was ~0.72
- New engineered features (e.g. url or hashtag count, text length, etc) while showing good correlation with the target, didn't add much to the performance of the model. Probably the reason is that this information was immanently added with frequency and polynomialization

## Examples of usage:

![img.png](img.png)
![img_1.png](img_1.png)