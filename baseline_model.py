import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
#https://www.kaggle.com/code/ludovicocuoghi/detecting-bullying-tweets-pytorch-lstm-bert/data

df = pd.read_csv("./cyberbullying_tweets.csv")


category_to_id={'not_cyberbullying':0, 'religion':1, 'age':2,'gender':3,'ethnicity':4,'other_cyberbullying':5}

print(df.head())
features= df.iloc[:,0].values
labels = df.iloc[:,1].values

tfidf=TfidfVectorizer(min_df=20,ngram_range=(1,2))
features = tfidf.fit_transform(features).toarray()
print(features)
print(features.shape)
print(labels.shape)

X_train, X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=6)

logreg = LogisticRegression()
logreg.fit(X_train, y_train) #fits model to logistic regression
y_test_pred=logreg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix) #shows confusion matrix
print("Accuracy is",accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred)) #shows accuracy metrics

