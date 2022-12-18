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

from datetime import datetime

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

import matplotlib.ticker as mticker

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
        range_results = range_k_grid_search.fit(X, y)
        default_results = default_grid_search.fit(X, y)
        
    return {
        'range': range_results,
        'default': default_results
    }

def k_range_scores(ranged_pipelines, cv=None, X=None, y=None):
    result = {}
    for k, rp in ranged_pipelines.items():
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Started {k} at {current_time}")
        result[k] = k_range_scores_for_pipe(rp[0], rp[1], cv=cv, X=X, y=y)
    return result
    

def svd_n_range_scores_for_pipe(pipe, n_range, no_poly_k=1000, poly_2_k=1000, defaults={}, cv=None, X=None, y=None):

    variable_defaults = {
        'No Poly, No BestK': {
            'poly2_k_best': ['passthrough']
         },
        f'No Poly, Best {no_poly_k}': {
            'poly2_k_best__poly2': ['passthrough'],
            'poly2_k_best__k_best__k': [no_poly_k]
        },
        f'Poly 2, Best {poly_2_k}': {
            'poly2_k_best__poly2': [PolynomialFeatures(2)],
            'poly2_k_best__k_best__k': [poly_2_k]
        },    
        f'Poly 2, Best 10000': {
            'poly2_k_best__poly2': [PolynomialFeatures(2)],
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
    result = {}
    for key, params in params_dict.items():
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Started {key} at {current_time}")
        result[key] = svd_n_range_scores_for_pipe(**params)
    return result

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
        
def plot_grid_results(results, x_var, legend_var=None, log=False):
    if legend_var is None:
        legend_values=['dummy']
    else:
        legend_values=results.param_grid[legend_var]
    for l_val in legend_values:
        scores = results.cv_results_['mean_test_score']
        x_values =  results.cv_results_[f'param_{x_var}'].data
        if legend_var:
            mask = [item[legend_var] == l_val for item in results.cv_results_['params']]
            scores = scores[mask]
            x_values = x_values[mask]
        if log:
            axes = plt.subplot()
            axes.set_xscale('log')
        plt.plot(x_values, scores, label=f'{legend_var} = {l_val}' if legend_var else '')
    plt.xlabel(x_var)
    plt.ylabel(results.scoring)
    if legend_var is not None:
        plt.legend()