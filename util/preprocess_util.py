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

def k_range_scores_for_pipe(pipe, k_range, cv=None, X=None, y=None):
    """
    Calculate scores for a given pipeline with k values from a range.
    
    Parameters:
    - pipe (sklearn Pipeline): The pipeline to score.
    - k_range (list): A list of integers representing the range of k values to test.
    - cv (int or sklearn model): The number of folds for cross-validation or a cross-validation splitter.
    - X (numpy array or pandas DataFrame): The input data.
    - y (numpy array or pandas Series): The target data.
    
    Returns:
    - dict: A dictionary with two keys: 'range' and 'default'. The 'range' key refers to a GridSearchCV object
            with k values from the provided range. The 'default' key refers to a GridSearchCV object without 
            either polynomialization or k best selection
    """
    param_grid_default = {
        'poly2_k_best': ['passthrough'],
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
    """
    Calculate scores for a given set of pipelines with k values from a range.
    
    Parameters:
    - ranged_pipelines (dict): A dictionary where the keys are strings representing the name of the pipeline and the
                               values are tuples with the first element being a sklearn Pipeline object and the second
                               element being a list of integers representing the range of k values to test.
    - cv (int or sklearn model): The number of folds for cross-validation or a cross-validation splitter.
    - X (numpy array or pandas DataFrame): The input data.
    - y (numpy array or pandas Series): The target data.
    
    Returns:
    - dict: A dictionary where the keys are the names of the pipelines and the values are dictionaries with two keys: 
            'range' and 'default'. The 'range' key refers to a GridSearchCV object with k values from the provided 
            range. The 'default' key refers to a GridSearchCV object  without either polynomialization or k best selection
    """    
    return {
        key: k_range_scores_for_pipe(rp[0], rp[1], cv=cv, X=X, y=y) for key, rp in ranged_pipelines.items()
    }

def svd_n_range_scores_for_pipe(pipe, n_range, no_poly_k=1000, poly_2_k=1000, defaults={}, cv=None, X=None, y=None):
    """
    Calculate scores for a given pipeline with a range of n values for the SVD step.
    
    Parameters:
    - pipe (sklearn Pipeline): The pipeline to score.
    - n_range (list): A list of integers representing the range of n values to test for the SVD step.
    - no_poly_k (int): The k value to use for the BestK step when the PolynomialFeatures step is set to 'passthrough'.
                      Default is 1000.
    - poly_2_k (int): The k value to use for the BestK step when the PolynomialFeatures step is set to PolynomialFeatures(2).
                      Default is 1000.
    - defaults (dict): A dictionary with parameter grids for the pipeline. Default is an empty dictionary.
    - cv (int or sklearn model): The number of folds for cross-validation or a cross-validation splitter.
    - X (numpy array or pandas DataFrame): The input data.
    - y (numpy array or pandas Series): The target data.
    
    Returns:
    - dict: A dictionary with keys representing the different parameter grid combinations being tested and values being
            dictionaries with two keys: 'range' and 'no_svd'. The 'range' key refers to a GridSearchCV object with 
            n values from the provided range for the SVD step. The 'no_svd' key refers to a GridSearchCV object with 
            the SVD step set to 'passthrough'.
    """
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
    """
    Calculate scores for a given set of pipelines with a range of n values for the SVD step.
    
    Parameters:
    - params_dict (dict): A dictionary where the keys are strings representing the name of the pipeline and the values 
                         are dictionaries with the parameters for the svd_n_range_scores_for_pipe function.
    
    Returns:
    - dict: A dictionary where the keys are the names of the pipelines and the values are dictionaries with keys 
            representing the different parameter grid combinations being tested and values being dictionaries with two 
            keys: 'range' and 'no_svd'. The 'range' key refers to a GridSearchCV object with n values from the 
            provided range for the SVD step. The 'no_svd' key refers to a GridSearchCV object with the SVD step set 
            to 'passthrough'.
    """    
    result = {}
    for key, params in params_dict.items():
        return {
            key: svd_n_range_scores_for_pipe(**params) for key, params in params_dict.items()
        }


def mislabeled_dups(df):
    """
    Find mislabeled duplicates in a dataframe.

    Mislabeled duplicates are records in the dataframe that have the same 'text' values, but different 'target' values.

    Parameters:
    - df (pandas.DataFrame): The dataframe to search for mislabeled duplicates.

    Returns:
    - pandas.DataFrame: A dataframe containing the mislabeled duplicates.
    """
    all_dups = df[df.duplicated(subset=['text'], keep=False)].text
    mislabeled_dups = all_dups[all_dups.apply(lambda x: df[df.text == x].target.nunique() == 2)]
    return mislabeled_dups


def drop_mislabeled_dups(df):
    """
    Drop mislabeled duplicates from a dataframe.

    Mislabeled duplicates are records in the dataframe that have the same 'text' values, but different 'target' values.

    Parameters:
    - df (pandas.DataFrame): The dataframe to search for and drop mislabeled duplicates.

    Returns:
    - pandas.DataFrame: A dataframe with the mislabeled duplicates removed.
    """
    return df.drop(mislabeled_dups(df).index)