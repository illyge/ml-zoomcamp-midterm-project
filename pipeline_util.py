from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2


def series_to_dict(series):
    """
    Convert a pandas Series object to a list of dictionaries with the series name as the key.
    
    Parameters:
    - series (pandas Series): The Series object to convert.
    
    Returns:
    - list: A list of dictionaries where each dictionary has a key with the name of the Series object.
    """    
    return [{series.name: item[1]} for item in series.items()]


def make_kw_pipeline():
    """
    Create a pipeline to convert a pandas Series object with keywords to a scipy sparse matrix.
    
    Returns:
    - sklearn Pipeline: A pipeline that takes in a pandas Series object and outputs a scipy sparse matrix.
    """
    return Pipeline(steps=[('kw_to_dict', FunctionTransformer(series_to_dict)),
                              ('d_vect', DictVectorizer())])

def make_loc_pipeline():
    """
    Create a pipeline to convert a pandas Series object with locations to a scipy sparse matrix.
    
    Returns:
    - sklearn Pipeline: A pipeline that takes in a pandas Series object and outputs a scipy sparse matrix.
    """    
    return Pipeline(steps=[('loc_to_dict', FunctionTransformer(series_to_dict)),
                              ('d_vect', DictVectorizer())])

def make_poly2_k_best_pipeline():
    """
    Create a pipeline to transform a scipy sparse matrix with PolynomialFeatures of degree 2 and apply SelectKBest using chi2.
    
    Returns:
    - sklearn Pipeline: A pipeline that takes in a scipy sparse matrix and outputs a transformed scipy sparse matrix.
    """    
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
    """
    Create a ColumnTransformer object that applies various vectorizers to different columns of a pandas DataFrame.
    
    Parameters:
    - min_df (int): The minimum number of documents a term must be in to be included in the vocabulary for text.
    - kw (bool): Whether to apply DictVectorizer to the 'keyword' column.
    - loc (bool): Whether to apply DictVectorizer to the 'location' column.
    - hashtags (bool): Whether to apply CountVectorizer to the 'hashtags' column.
    - urls (bool): Whether to apply CountVectorizer to the 'urls' column.
    - stopwords (list): A list of words to ignore when tokenizing the text column.
    - preprocessor (callable): A function to apply to the text column before tokenizing.
    - ngram_range (tuple): The range of n-grams to consider when tokenizing the text column.
    
    Returns:
    - sklearn ColumnTransformer: A ColumnTransformer object that applies the specified vectorizers to the specified columns of a pandas DataFrame.
    """    
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
    """
    Constructs a preparation pipeline for text data.

    Parameters
    - url_cleaner : a transformer object or None (default=None)
    A transformer object to be applied to the 'urls' column in the input data.

    - min_df : int (default=10)
    The minimum number of documents a word must be in to be included in the vocabulary of the CountVectorizer applied to the 'text' column.

    - kw : bool (default=False)
    Whether to include a pipeline that processes the 'keyword' column in the input data.

    - loc : bool (default=False)
    Whether to include a pipeline that processes the 'location' column in the input data.

    - hashtags : bool (default=False)
    Whether to include a pipeline that processes the 'hashtags' column in the input data.

    - urls : bool (default=False)
    Whether to include a pipeline that processes the 'urls' column in the input data.

    - stopwords : list or None (default=None)
    A list of words to be excluded from the vocabulary of the CountVectorizer applied to the 'text' column.

    - preprocessor : a callable or None (default=None)
    A callable that preprocesses the 'text' column in the input data.

    - ngram_range : tuple (default=(1, 1))
    The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

    Returns
    - A scikit-learn pipeline object that processes the input data and returns a transformed version of it, ready for modeling.
    """
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
    """
    Constructs a pipeline with the given classifier at the end and a variety of optional preprocessing steps, specified in params, before it.

    Parameters:
    - classifier: a scikit-learn classifier object
    The final step in the pipeline, which will be fit and used to make predictions on the transformed data.
    - params: dict
    A dictionary of parameters for preprocessing the data. Possible keys are:
    - url_cleaner: A transformer object to clean the 'urls' column
    - min_df: The minimum number of documents a word must be in to be included in the vocabulary of the CountVectorizer (default 10)
    - kw: Boolean, whether to include the 'keyword' column in the final feature matrix (default False)
    - loc: Boolean, whether to include the 'location' column in the final feature matrix (default False)
    - hashtags: Boolean, whether to include the 'hashtags' column in the final feature matrix (default False)
    - urls: Boolean, whether to include the 'urls' column in the final feature matrix (default False)
    - stopwords: A list of words to exclude from the vocabulary of the CountVectorizer (default None)
    - preprocessor: A function to preprocess the 'text' column (default None)
    - ngram_range: A tuple specifying the range of n-grams to include in the final feature matrix (default (1, 1))

    Returns:
    - A scikit-learn Pipeline object with the specified preprocessing steps and the given classifier at the end.
    """
    
    pipeline = make_preparation_pipeline(**params)
    steps = pipeline.steps.copy()
    steps.append(('poly2_k_best', make_poly2_k_best_pipeline()))
    steps.append(('svd', TruncatedSVD(random_state=42)))
    steps.append(('classifier', classifier))

    return Pipeline(steps)