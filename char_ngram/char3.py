from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import json
from utils import load_twitter_dataset, load_setimes_dataset
from collections import Counter

clf=SGDClassifier()

twitter_X,twitter_y=load_twitter_dataset()
twitter3_X,twitter3_y=load_twitter_dataset(['bs','hr','sr'])
setimes_X,setimes_y=load_setimes_dataset()

clf = Pipeline([
    ('vect', CountVectorizer(analyzer="char",ngram_range=(3,3))),
    ('clf', clf)
])

clf.fit(setimes_X['train'],setimes_y['train'])
print('setimes clf on setimes_test')
print(classification_report(setimes_y['test'],clf.predict(setimes_X['test'])))
print('setimes clf on twitter3_test')
print(classification_report(twitter3_y['test'],clf.predict(twitter3_X['test'])))

clf.fit(twitter3_X['train'],twitter3_y['train'])
print('twitter3 clf on setimes_test')
print(classification_report(setimes_y['test'],clf.predict(setimes_X['test'])))
print('twitter3 clf on twitter_test')
print(classification_report(twitter3_y['test'],clf.predict(twitter3_X['test'])))

clf.fit(twitter_X['train'],twitter_y['train'])
print('twitter clf on twitter_test')
print(classification_report(twitter_y['test'],clf.predict(twitter_X['test'])))