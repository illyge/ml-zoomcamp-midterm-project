import bentoml
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from pipeline_util import make_preparation_pipeline
from sklearn.pipeline import Pipeline
from preprocess_util import drop_mislabeled_dups

def pipeline():
    pipeline = make_preparation_pipeline(kw=True)
    steps = pipeline.steps
    steps.append(('classifier', ComplementNB()))
    return Pipeline(steps)


def prepare_data(df):
    df = df.fillna("")
    df = drop_mislabeled_dups(df)
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
