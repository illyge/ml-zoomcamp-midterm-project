import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
import logging

from sklearn.feature_selection import SelectKBest, chi2
default_preprocess_settings = {
    'url_cleaner': None,
    'preprocessor': None,
    'stopwords': None,
    'ngram_range': (1, 1),
    'best_k': None,
    'max_poly': 1,
    'kw': False,
    'loc': False,
    'hashtags': False,
    'svd_n_comp': None
}

def prepare_train_data(df, y, settings = default_preprocess_settings):
    
    url_cleaner, preprocessor, stopwords, ngram_range, best_k, max_poly, kw, loc, hashtags, svd_n_comp = itemgetter(
        'url_cleaner',
        'preprocessor',
        'stopwords',
        'ngram_range',
        'best_k',
        'max_poly',
        'kw',
        'loc',
        'hashtags',
        'svd_n_comp')(settings)
    
    min_df = 5
    
    if stopwords and preprocessor:
        stopwords = [preprocessor(word) for word in stopwords]
    
    c_vectorizer = CountVectorizer(min_df = min_df,
                                   ngram_range = ngram_range,
                                   preprocessor = preprocessor,
                                   stop_words = stopwords)
    kw_d_vectorizer = None
    loc_d_vectorizer = None
    p_features = PolynomialFeatures(max_poly)
    s_k_best = None
    ht_c_vectorizer = None
    svd = None
    
    text = df.text

    if url_cleaner:
        text = url_cleaner.transform(df).text
        
    vectors = c_vectorizer.fit_transform(text)

    if kw:
        kw_d_vectorizer = DictVectorizer()
        train_kw_dict = df[['keyword']].to_dict(orient='records')
        train_kw_vectors = kw_d_vectorizer.fit_transform(train_kw_dict)
        vectors = sp.hstack((vectors, train_kw_vectors))

    if loc:
        loc_d_vectorizer = DictVectorizer()
        train_loc_dict = df[['location']].to_dict(orient='records')
        train_loc_vectors = loc_d_vectorizer.fit_transform(train_loc_dict)
        vectors = sp.hstack((vectors, train_loc_vectors))
        
    if hashtags:
        ht_c_vectorizer = CountVectorizer(min_df = 5, analyzer = lambda x: x)
        ht_vectors = ht_c_vectorizer.fit_transform(df.hashtags)
        vectors = sp.hstack((vectors, ht_vectors))
        
    vectors =  p_features.fit_transform(vectors)

    if best_k:
        s_k_best = SelectKBest(chi2, k=min(vectors.shape[1], best_k))
        vectors = s_k_best.fit_transform(vectors, y)
    

    if svd_n_comp:
        svd = TruncatedSVD(n_components=svd_n_comp, n_iter=20, random_state=42)
        vectors = svd.fit_transform(vectors)
        
    return {
        'settings': settings,
        'transformers': {
            'url_cleaner': url_cleaner,
            'c_vect': c_vectorizer,
            'kw_d_vect': kw_d_vectorizer,
            'loc_d_vect': loc_d_vectorizer,
            'p_feat': p_features,
            's_k_best': s_k_best,
            'ht_c_vect': ht_c_vectorizer,
            'svd': svd
        },
        'vectors': vectors
    }

def prepare_test_vectors(df, train_data):
    url_cleaner, c_vectorizer, kw_d_vectorizer, loc_d_vectorizer, p_features, s_k_best, ht_c_vectorizer, svd = itemgetter(
        'url_cleaner',
        'c_vect',
        'kw_d_vect',
        'loc_d_vect',
        'p_feat',
        's_k_best',
        'ht_c_vect',
        'svd')(train_data['transformers'])

    text = df.text
    if url_cleaner:
        text = url_cleaner.transform(df).text
        
    vectors = c_vectorizer.transform(text)
    
    if kw_d_vectorizer:
        kw_dict = df[['keyword']].to_dict(orient='records')
        kw_vectors = kw_d_vectorizer.transform(kw_dict)
        vectors = sp.hstack((vectors, kw_vectors))

    if loc_d_vectorizer:
        loc_dict = df[['location']].to_dict(orient='records')
        loc_vectors = loc_d_vectorizer.transform(loc_dict)
        vectors = sp.hstack((vectors, loc_vectors))
        
    if ht_c_vectorizer:
        ht_vectors = ht_c_vectorizer.transform(df.hashtags)
        vectors = sp.hstack((vectors, ht_vectors))
        
    vectors =  p_features.transform(vectors)

    if s_k_best:
        vectors = s_k_best.transform(vectors)
    
    if svd:
        vectors = svd.transform(vectors)
        
    return vectors

def prepare_cross_val_vectors(X, y, settings=default_preprocess_settings, cv=None):
    splits = []
    for train_index, val_index in cv.split(X, y):
        y_train = y.iloc[train_index]
        train_data = prepare_train_data(X.iloc[train_index], y_train, settings=settings)

        X_val = prepare_test_vectors(X.iloc[val_index], train_data)
        y_val = y.iloc[val_index]
        splits.append({
            'X_train': train_data['vectors'],
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'transformers': train_data['transformers']
        })
    return {
        'splits': splits,
        'settings': settings
    }

def my_cross_val_score(model, splits):
    scores = []
    for split in splits:
        model.fit(split['X_train'], split['y_train'])
        scores.append(f1_score(split['y_val'], model.predict(split['X_val'])))
    return np.array(scores)

def compare_models(models, vectors_combinations):
    df = pd.DataFrame()
    for key, data in vectors_combinations.items():
        record_dict = {
            'Features': key
        }
        for model_key, model in models.items():
            try:
                mean_score = my_cross_val_score(model, data['splits']).mean()
            except Exception as e:
                logging.warning(e)
                mean_score = None
            finally:
                record_dict[model_key] = mean_score

        df = pd.concat([df, pd.DataFrame([record_dict])])
    df.set_index('Features', inplace=True)
    return df
