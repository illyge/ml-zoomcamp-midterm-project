import bentoml
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.sparse as sp
from sklearn.feature_selection import SelectKBest, chi2


def prepare_data(df, k_best=20000, svd_n_components=1000):
    min_df = 30
    c_vectorizer = CountVectorizer(min_df=min_df)
    d_vectorizer = DictVectorizer()

    kw_loc_dict = df[['keyword', 'location']].to_dict(orient='records')

    kw_loc_vectors = d_vectorizer.fit_transform(kw_loc_dict)
    text_vectors = c_vectorizer.fit_transform(df.text)

    text_kw_loc_vectors = sp.hstack((text_vectors, kw_loc_vectors))
    p_features = PolynomialFeatures(2)
    text_kw_loc_poly_vectors = p_features.fit_transform(text_kw_loc_vectors)
    s_k_best = SelectKBest(chi2, k=k_best)
    best_vectors = s_k_best.fit_transform(text_kw_loc_poly_vectors, df.target)

    svd = TruncatedSVD(n_components=svd_n_components, n_iter=20, random_state=42)
    vectors = svd.fit_transform(best_vectors)

    return (vectors, {
        'c_vect': c_vectorizer,
        'd_vect': d_vectorizer,
        'p_feat': p_features,
        's_k_best': s_k_best,
        'svd': svd,
        'vectors': vectors
    })


def train():
    data = pd.read_csv("train.csv")
    data = data.fillna("")
    train_data = prepare_data(data)
    model = LogisticRegression(solver="sag", C=1.0, max_iter=1000, random_state=42)
    model.fit(train_data[0], data.target)

    print(bentoml.sklearn.save_model(
        'twitter-disasters-model',
        model,
        custom_objects=train_data[1]
    ))


if __name__ == '__main__':
    train()
