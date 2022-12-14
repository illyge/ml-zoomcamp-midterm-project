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
import warnings

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

from sklearn.model_selection import GridSearchCV

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
        
    # vectors =  p_features.fit_transform(vectors)

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
        
    # vectors =  p_features.transform(vectors)

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




def plot_score_diffs(diffs):
    fig, ax = plt.subplots(6, 1, figsize=(10,35))

    diffs_reversed = diffs.iloc[::-1]
    steps_range = np.arange(diffs_reversed.shape[0])

    for i, col in enumerate(diffs.columns):
        diffs_sorted = diffs.sort_values(by=[col])
        steps_range = np.arange(diffs_sorted.shape[0])

        axes = plt.subplot(6, 1, i+1) 
        plt.title(col)

        scores = diffs_sorted[col].to_list()

        positive_scores = [max(0, s) for s in scores]
        negative_scores = [abs(min(0, s)) for s in scores]

        features = list(diffs_sorted.index)

        ticks = [f"{feature} ({round(score, 3)})" for feature, score in zip(features, scores)]

        plt.barh(steps_range, width=positive_scores, height=0.7, label='Positive difference', color='green')
        plt.barh(steps_range, width=negative_scores, height=0.7, label='Negative difference', color='red')

        plt.yticks(steps_range, ticks)
        plt.legend()
        plt.ylabel('Preparation steps')
        plt.xlabel('F1 Score difference vs Pure Text')

    plt.legend()

def k_range_scores_for_pipe(pipe, k_range, cv=None, X=None, y=None):
    print (f"Doing another pipe")
    
    param_grid_default = {
        'poly2_k_best': [None],
        'svd': ['passthrough']
    }
    
    param_grid_k_range = {
        'poly2_k_best__poly2': ['passthrough', PolynomialFeatures(2)],
        'poly2_k_best__k_best__k': k_range,
        'svd': ['passthrough']
    }
    
    range_k_grid_search = GridSearchCV(pipe, 
                                   param_grid = param_grid_k_range,
                                   scoring='f1',
                                   cv=cv)
    default_grid_search = GridSearchCV(pipe, 
                                   param_grid = param_grid_default,
                                   scoring='f1',
                                   cv=cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('fitting range')
        range_results = range_k_grid_search.fit(X, y)
        print('fitting default')
        default_results = default_grid_search.fit(X, y)
        
    return {
        'range': range_results,
        'default': default_results
    }

def k_range_scores(ranged_pipelines, cv=None, X=None, y=None):
    print("whatever")
    return {
        k: k_range_scores_for_pipe(rp[0], rp[1], cv=cv, X=X, y=y) for k, rp in ranged_pipelines.items()
    } 

def svd_n_range_scores_for_pipe(pipe, n_range, no_poly_k=1000, poly_2_k=1000, defaults={}, cv=None, X=None, y=None):
    variable_defaults = {
        'Pure Text': {
            'poly2_k_best': ['passthrough']
         },
        f'No Poly, Best {no_poly_k}': {
            'poly2_k_best__poly2': ['passthrough'],
            'poly2_k_best__k_best__k': [no_poly_k]
        },
        f'Poly 2, Best {poly_2_k}': {
            'poly2_k_best__k_best__k': [poly_2_k]
        },    
        f'Poly 2, Best 10000': {
            'poly2_k_best__k_best__k': [10000]
        }         
    }
    
    results = {}
    for label, param_grid in variable_defaults.items():
        param_grid_ranged = defaults.copy()
        param_grid_ranged.update(param_grid)
        param_grid_no_svd = param_grid_ranged.copy() 
        param_grid_ranged.update({
            'svd__n_components': n_range
        })
        param_grid_no_svd.update({
            'svd': ['passthrough']
        })
            
        # print(pipe)
        # print(param_grid_ranged)
        # pring(cv)
        grid_search = GridSearchCV(pipe, 
                                   param_grid = param_grid_ranged,
                                   scoring='f1',
                                   cv=cv)
        
        grid_search_no_svd = GridSearchCV(pipe, 
                                       param_grid = param_grid_no_svd,
                                       scoring='f1',
                                       cv=cv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            range_results = grid_search.fit(X, y)
            no_svd_results = grid_search_no_svd.fit(X, y)
        results[label] = {
            'range': range_results,
            'no_svd': no_svd_results
        }
        
        
    return results

def svd_n_range_scores(params_dict):
    return {
        key: svd_n_range_scores_for_pipe(**params) for key, params in params_dict.items()
    }

import matplotlib.ticker as mticker

def plot_k_range_results(all_results):
    fig, ax = plt.subplots(len(all_results), 1, figsize=(10,35))

    i=1
    for k, results in all_results.items():
        axes = plt.subplot(6, 1, i) 
        i+=1
        plt.title(k)
        
        axes.set_xscale('log')
        axes.xaxis.set_minor_formatter(mticker.ScalarFormatter())

        range_results = results['range'].cv_results_
        default_results = results['default'].cv_results_

        non_poly_mask = [item['poly2_k_best__poly2'] == 'passthrough' for item in range_results['params']]
        poly2_mask = [not i for i in non_poly_mask]
        non_poly_scores = range_results['mean_test_score'][non_poly_mask]
        poly2_scores = range_results['mean_test_score'][poly2_mask]
        k_range = range_results['param_poly2_k_best__k_best__k'][poly2_mask]

        default_score = default_results['mean_test_score'][0]

        plt.plot(k_range.data, poly2_scores, label='With poly2', color='blue')
        max_poly2 = max(zip(k_range.data, poly2_scores), key=lambda x:x[1])
        plt.axhline(y = max_poly2[1], color = 'blue', linestyle = ':', label=f"Max poly 2 {round(max_poly2[1], 3)} at {max_poly2[0]}")

        plt.plot(k_range.data, non_poly_scores, label='Without poly', color='orange')
        max_non_poly = max(zip(k_range.data, non_poly_scores), key=lambda x:x[1])
        plt.axhline(y = max_non_poly[1], color = 'orange', linestyle = ':', label=f"Max non poly {round(max_non_poly[1], 3)} at {max_non_poly[0]}")

        plt.axhline(y = default_score, color = 'gray', linestyle = ':', label=f"Default {round(default_score, 3)}")

        plt.xlabel('k in SelectKBest')
        plt.ylabel('F1 score')

        plt.legend()
        print(k_range.data)
    
def plot_svd__range_results(all_results):
    fig, ax = plt.subplots(len(all_results), 1, figsize=(10,35))

    i=1    
    
    for key, results in all_results.items():
        axes = plt.subplot(6, 1, i) 
        i+=1
        plt.title(key)
        
        # figure(figsize=(10, 7), dpi=80)
        color = iter(cm.rainbow(np.linspace(0, 1, len(results))))

        for label, labeled_results in results.items():
            c = next(color)

            range_results = labeled_results['range'].cv_results_
            no_svd_results = labeled_results['no_svd'].cv_results_

            svd_n_range = range_results['param_svd__n_components'].data
            scores = range_results['mean_test_score']


            no_svd_score = no_svd_results['mean_test_score'][0]

            plt.plot(svd_n_range, scores, label=label, color=c)
            max_score = max(zip(svd_n_range, scores), key=lambda x:x[1])

            plt.axhline(y = max_score[1], color = c, linestyle = ':', label=f"Max score for {label} {round(max_score[1], 3)} at {max_score[0]}")
            plt.axhline(y = no_svd_score, color = c, linestyle = '--', label=f"No SVD {round(no_svd_score, 3)}")

        plt.xlabel('n components in TruncatedSVD')
        plt.ylabel('F1 score')
        plt.legend()