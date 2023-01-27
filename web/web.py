from utils import transliterate, load_fasttext, get_stats
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import json
from utils import load_twitter_dataset, load_setimes_dataset
from collections import Counter
from random import seed
seed(42)
import pandas as pd

twitter_X,twitter_y=load_twitter_dataset()
twitter3_X,twitter3_y=load_twitter_dataset(['bs','hr','sr'])
setimes_X,setimes_y=load_setimes_dataset()
twitter3_train = pd.DataFrame({"labels": twitter3_y["train"], "text": twitter3_X["train"]})
twitter3_test = pd.DataFrame({"labels": twitter3_y["test"], "text": twitter3_X["test"]})

twitter_train = pd.DataFrame({"labels": twitter_y["train"], "text": twitter_X["train"]})
twitter_test = pd.DataFrame({"labels": twitter_y["test"], "text": twitter_X["test"]})

setimes_train = pd.DataFrame({"labels": setimes_y["train"], "text": setimes_X["train"]})
setimes_test = pd.DataFrame({"labels": setimes_y["test"], "text": setimes_X["test"]})




pred_report={'system':'web+NaiveBayes','predictions':[]}

preds=pred_report['predictions']
print('setimes clf on setimes_test')
d = get_stats(
    train_df = setimes_train,
    eval_df = setimes_test,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)[0]
pred=d["y_pred"]
preds.append({'train':'setimes','test':'setimes','predictions':pred })
print(classification_report(setimes_y['test'],pred,digits=3))

print('setimes clf on twitter3_test')
d = get_stats(
    train_df = setimes_train,
    eval_df = twitter3_test,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)[0]
pred=d["y_pred"]
preds.append({'train':'setimes','test':'twitter3','predictions':pred })
print(classification_report(twitter3_y['test'],pred,digits=3))


print('twitter3 clf on setimes_test')
d = get_stats(
    train_df = twitter3_train,
    eval_df = setimes_test,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)[0]
pred=d["y_pred"]
preds.append({'train':'twitter3','test':'setimes','predictions':pred })
print(classification_report(setimes_y['test'],pred,digits=3))
print('twitter3 clf on twitter_test')
d = get_stats(
    train_df = twitter3_train,
    eval_df = twitter_test,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)[0]
pred=d["y_pred"]
preds.append({'train':'twitter3','test':'twitter3','predictions':pred })
print(classification_report(twitter_y['test'],pred,digits=3))


print('twitter clf on twitter_test')
d = get_stats(
    train_df = twitter_train,
    eval_df = twitter_test,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)[0]
pred=d["y_pred"]
preds.append({'train':'twitter','test':'twitter','predictions':pred })
print(classification_report(twitter_y['test'],pred,digits=3))

json.dump(pred_report,open('web.predictions.json','wt'))
