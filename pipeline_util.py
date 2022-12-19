from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2

def kw_to_dict(series):
    return [{'keyword': item[1]} for item in series.items()]

def loc_to_dict(series):
    return [{'location': item[1]} for item in series.items()]

def make_kw_pipeline():
    return Pipeline(steps=[('kw_to_dict', FunctionTransformer(kw_to_dict)),
                              ('d_vect', DictVectorizer())])
def make_loc_pipeline():
    return Pipeline(steps=[('loc_to_dict', FunctionTransformer(loc_to_dict)),
                              ('d_vect', DictVectorizer())])

def make_poly2_k_best_pipeline():
    return Pipeline(steps=[('poly2', PolynomialFeatures(2)),
                                        ('k_best', SelectKBest(chi2))])

def make_vectorizer(
        min_df=10,
        kw=False,
        loc=False,
        hashtags=False,
        urls=False,
        stopwords=None,
        preprocessor=None,
        ngram_range=(1, 1)):
    columns = [('text_c_vect', CountVectorizer(min_df=min_df, stop_words=stopwords, preprocessor=preprocessor, ngram_range=ngram_range), 'text')]
    if kw:
        columns.append(('kw_dict_vect', make_kw_pipeline(), 'keyword'))
    if loc:
        columns.append(('loc_dict_vect', make_loc_pipeline(), 'location'))
    if hashtags:
        columns.append(('hashtags_c_vect', CountVectorizer(min_df=5, analyzer=lambda x: x), 'hashtags'))
    if urls:
        columns.append(('urls_c_vect', CountVectorizer(min_df=3, analyzer=lambda x: x), 'urls'))

    return ColumnTransformer(columns)


def make_preparation_pipeline(
        url_cleaner=None,
        min_df=10,
        kw=False,
        loc=False,
        hashtags=False,
        urls=False,
        stopwords=None,
        preprocessor=None,
        ngram_range=(1, 1),
    ):

    vectorizer=make_vectorizer(
        min_df=min_df,
        kw=kw,
        loc=loc,
        urls=urls,
        hashtags=hashtags,
        stopwords=stopwords,
        preprocessor=preprocessor,
        ngram_range=ngram_range)

    steps = [('url_cleaner', url_cleaner)]
    steps.append(('vectorizer', vectorizer))

    return Pipeline(steps=steps)

def make_transformation_pipeline(
    classifier,
    params
):
    pipeline = make_preparation_pipeline(**params)
    steps = pipeline.steps.copy()
    steps.append(('poly2_k_best', make_poly2_k_best_pipeline()))
    steps.append(('svd', TruncatedSVD(algorithm='arpack')))
    steps.append(('classifier', classifier))

    return Pipeline(steps)