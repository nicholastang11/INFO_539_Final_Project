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

#['not_cyberbullying',  'religion',  'age',  'gender','ethnicity', 'other_cyberbullying']
category_to_id={'not_cyberbullying':0, 'religion':1, 'age':2,'gender':3,'ethnicity':4,'other_cyberbullying':5}
#category_to_id={'not_cyberbullying':0, 'religion':1, 'age':2,'gender':3,'ethnicity':4,'other_cyberbullying':5}
#df['cyberbullying_type'].replace(category_to_id, inplace=True)
print(df.head())
features= df.iloc[:,0].values
labels = df.iloc[:,1].values

#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=25,max_df=.8, norm='l2',use_idf=True,lowercase=True,
# encoding='latin-1', stop_words='english',strip_accents='unicode',smooth_idf=True)
tfidf=TfidfVectorizer(min_df=20,ngram_range=(1,2))
features = tfidf.fit_transform(features).toarray()
print(features)
print(features.shape)
print(labels.shape)


from sklearn.feature_selection import chi2
N = 2
'''
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
'''
#random state 3 is 83.1% accuracy
X_train, X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=6)
#clf=LinearSVC(random_state=0)
#clf.fit(X_train,y_train)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_test_pred=logreg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix)
print("Accuracy is",accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
#for input, prediction, label in zip (X_test, y_test_pred, y_test):
#  if prediction != label:
#    print(input, 'has been classified as ', prediction, 'and should be ', label)
