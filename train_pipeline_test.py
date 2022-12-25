import bentoml
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from pipeline_util import make_preparation_pipeline
from sklearn.pipeline import Pipeline


def pipeline():
    pipeline = make_preparation_pipeline(kw=True)
    steps = pipeline.steps
    steps.append(('classifier', ComplementNB()))
    return Pipeline(steps)


def prepare_data(df):
    df = df.fillna("")
    all_dups = df[df.duplicated(subset=['text'], keep=False)].text
    mislabeled_dups = all_dups[all_dups.apply(lambda x: df[df.text == x].target.nunique() == 2)]
    df = df.drop(mislabeled_dups.index)
    return df


def train():
    data = pd.read_csv("train.csv")
    train_data = prepare_data(data)

    classifier = pipeline()
    classifier.fit(train_data, train_data.target)

    print(bentoml.sklearn.save_model(
        'twitter-disasters-model',
        classifier
    ))


if __name__ == '__main__':
    train()
