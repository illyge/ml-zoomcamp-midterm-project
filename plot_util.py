import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib.ticker as mticker

def plot_score_diffs(diffs):
    """
    Plot horizontal bar charts comparing F1 scores for different preparation steps.
    
    The bars are colored green for positive differences in F1 score and red for negative differences. The difference in F1 score for each preparation step is also displayed next to the step name.
    
    Parameters:
    - diffs (pandas DataFrame): A DataFrame with preparation steps as the index and F1 score differences as the values. The DataFrame should have 6 columns, corresponding to the 6 different F1 scores being plotted.
    
    Returns:
    - None
    """    
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

def plot_k_range_results(all_results):
    """
    Plot the results of grid search for different values of k in SelectKBest.
    
    The plots compare the F1 scores for models with and without the use of the Poly2 transformer, as well as the F1 score for the default configuration.
    
    Parameters:
    - all_results (dict): A dictionary with keys corresponding to the values of k being tested and values corresponding to dictionaries with the results of grid search for 'range' and 'default' configurations.
    
    Returns:
    - None
    """
    
    fig, ax = plt.subplots(len(all_results), 1, figsize=(10, 35))

    i = 1
    for k, results in all_results.items():
        axes = plt.subplot(6, 1, i)
        i += 1
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
        max_poly2 = max(zip(k_range.data, poly2_scores), key=lambda x: x[1])
        plt.axhline(y=max_poly2[1], color='blue', linestyle=':',
                    label=f"Max poly 2 {round(max_poly2[1], 3)} at {max_poly2[0]}")

        plt.plot(k_range.data, non_poly_scores, label='Without poly', color='orange')
        max_non_poly = max(zip(k_range.data, non_poly_scores), key=lambda x: x[1])
        plt.axhline(y=max_non_poly[1], color='orange', linestyle=':',
                    label=f"Max non poly {round(max_non_poly[1], 3)} at {max_non_poly[0]}")

        plt.axhline(y=default_score, color='gray', linestyle=':', label=f"Default {round(default_score, 3)}")

        plt.xlabel('k in SelectKBest')
        plt.ylabel('F1 score')

        plt.legend()


def plot_svd__range_results(all_results):
    """
    Plot the results of grid search for different values of n_components in TruncatedSVD.
    
    The plots compare the F1 scores for models with and without the use of TruncatedSVD, as well as the F1 score for each value of n_components.
    
    Parameters:
    - all_results (dict): A dictionary with keys corresponding to the values of k being tested and values corresponding to dictionaries with labels for each set of results. The values for these labels are dictionaries with the results of grid search for 'range' and 'no_svd' configurations.
    
    Returns:
    - None
    """    
    fig, ax = plt.subplots(len(all_results), 1, figsize=(10, 35))

    i = 1

    for key, results in all_results.items():
        axes = plt.subplot(6, 1, i)
        i += 1
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
            max_score = max(zip(svd_n_range, scores), key=lambda x: x[1])

            plt.axhline(y=max_score[1], color=c, linestyle=':',
                        label=f"Max score for {label} {round(max_score[1], 3)} at {max_score[0]}")
            plt.axhline(y=no_svd_score, color=c, linestyle='--', label=f"No SVD {round(no_svd_score, 3)}")

        plt.xlabel('n components in TruncatedSVD')
        plt.ylabel('F1 score')
        plt.legend()


def plot_grid_results(results, x_var, legend_var=None, log=False):
    """
    Plot the results of a grid search as a function of one of the parameters.
    
    Parameters:
    - results (GridSearchCV): The results of the grid search.
    - x_var (str): The name of the parameter to plot on the x-axis.
    - legend_var (str, optional): The name of the parameter to use in the legend. If not provided, no legend will be shown.
    - log (bool, optional): Whether to plot the x-axis on a log scale.
    
    Returns:
    - None
    """    
    if legend_var is None:
        legend_values = ['dummy']
    else:
        legend_values = results.param_grid[legend_var]
    for l_val in legend_values:
        scores = results.cv_results_['mean_test_score']
        x_values = results.cv_results_[f'param_{x_var}'].data
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