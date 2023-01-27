

# %%
import pandas as pd
from pathlib import Path
from utils import transliterate

setimes = pd.read_json(str(Path("repodata/data/SETimes.HBS.json"))).rename(columns={"language": "labels"})
twitter = pd.read_json(str(Path("repodata/data/Twitter-HBS.json"))).rename(columns={"language":  "labels"})
twitter["text"] = twitter.tweets.apply(lambda l: " ".join(l))


url_pattern = r"[A-Za-z]+:\/\/[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_:%&;\?\#\/.=]+"
username_pattern = r"@\S+"
hashtag_pattern = r"#\S+"
twitter["text"] = twitter.text.str.replace(
    url_pattern, "", regex=True).str.replace(
    username_pattern, " ", regex=True).str.replace(
    hashtag_pattern, "", regex=True ).str.replace(
    "\sRT\s+:", " ", regex=True).apply(transliterate).str.replace(
    "\n", " ")

# %%
from utils import load_fasttext, get_stats
wac100 = load_fasttext(str(Path("ndat/final/wacs_100k.fasttext")))

resultdict, vec, clf = get_stats(
    train_df = wac100,
    eval_df = setimes,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)

# %%
y_true, y_pred = resultdict["y_true"], resultdict["y_pred"]
y_true == setimes.labels.tolist()
setimes["y_pred"] = y_pred

# %%

resultdict, vec, clf = get_stats(
    train_df = wac100[wac100.labels!="me"],
    eval_df = twitter[twitter.labels!="me"],
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)

# %%
twitter["twitter_3"] = None
twitter.loc[twitter.labels!="me", "twitter_3"] = resultdict["y_pred"]

# %%
resultdict, vec, clf = get_stats(
    train_df = wac100,
    eval_df = twitter,
    classifier_type="NaiveBayes",
    vectorizer_type="web",
)

twitter["twitter_4"] = resultdict["y_pred"]

# %%
results = [
    {"setimes": setimes.y_pred.tolist(),
     "twitter_3": twitter.twitter_3.tolist(),
     "twitter_4": twitter.twitter_4.tolist()}
]
import json
with open(Path("hbs-lid-benchmark/web/results.json").__str__(), "w") as f:
    json.dump(results, f)


