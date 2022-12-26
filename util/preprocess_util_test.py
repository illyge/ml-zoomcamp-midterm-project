from preprocess_util import k_range_scores_for_pipe, \
    svd_n_range_scores_for_pipe, \
    mislabeled_dups, \
    drop_mislabeled_dups

import unittest
from unittest.mock import Mock, patch
import pandas as pd


class TestFunctions(unittest.TestCase):
    
    @patch('preprocess_util.GridSearchCV', autospec=True)
    @patch('preprocess_util.PolynomialFeatures', autospec=True)
    def test_k_range_scores_for_pipe(self, mock_polynomial_features, mock_grid_search):
        # Create a mock pipeline object
        mock_pipeline = Mock()

        # Create a mock CV object
        mock_cv = Mock()

        # Set up the mock object to return a predefined value when the fit method is called
        mock_pipeline.fit.return_value = mock_pipeline

        # Set up the mock object to have a default value for the `best_params_` attribute
        mock_pipeline.best_params_ = {'param1': 1, 'param2': 2}

        # Set up the mock for Polynomial Features
        mock_polynomial_features_rv = Mock()
        mock_polynomial_features.return_value = mock_polynomial_features_rv

        # Set up the mock grid search object to return itself when the fit method is called
        mock_grid_search.return_value = mock_grid_search
        
        mock_grid_search_fit = Mock()
        mock_grid_search.fit.return_value = mock_grid_search_fit
        # Create mock objects for the X and y inputs
        mock_X = Mock()
        mock_y = Mock()

        # Call the function with the mock objects
        result = k_range_scores_for_pipe(mock_pipeline, [1, 2, 3], cv=mock_cv, X=mock_X, y=mock_y)

        # Assert that the function returns the expected result
        self.assertIsInstance(result, dict)
        self.assertIn('range', result)
        self.assertIn('default', result)
        self.assertIs(result['range'], mock_grid_search_fit)
        self.assertIs(result['default'], mock_grid_search_fit)

        # Assert that the mock grid search object was called twice
        self.assertEqual(mock_grid_search.call_count, 2)


        # Assert that the first call to the mock grid search object used the correct parameters

        self.assertEqual(mock_grid_search.call_args_list[0][0], (mock_pipeline,))
        self.assertEqual(mock_grid_search.call_args_list[0][1],
                         {'param_grid':
                              {'poly2_k_best__poly2': ['passthrough', mock_polynomial_features_rv],
                               'poly2_k_best__k_best__k': [1, 2, 3],
                               'svd': ['passthrough']},
                          'scoring': 'f1',
                          'cv': mock_cv
                          })

        # Assert that the second call to the mock grid search object used the correct parameters

        self.assertEqual(mock_grid_search.call_args_list[1][0], (mock_pipeline,))
        self.assertEqual(mock_grid_search.call_args_list[1][1],
                         {'param_grid':
                              {'poly2_k_best': ['passthrough'],
                               'svd': ['passthrough']},
                          'scoring': 'f1',
                          'cv': mock_cv
                          })

        # Assert that both calls to the fit method used the correct parameters
        self.assertEqual(mock_grid_search.fit.call_args_list[0], ((mock_X, mock_y),))
        self.assertEqual(mock_grid_search.fit.call_args_list[1], ((mock_X, mock_y),))


    @patch('preprocess_util.GridSearchCV', autospec=True)
    @patch('preprocess_util.PolynomialFeatures', autospec=True)
    def test_svd_n_range_scores_for_pipe(self, mock_polynomial_features, mock_grid_search):
        # Create a mock pipeline object
        mock_pipeline = Mock()

        # Create a mock CV object
        mock_cv = Mock()

        # Set up the mock object to return a predefined value when the fit method is called
        mock_pipeline.fit.return_value = mock_pipeline

        # Set up the mock object to have a default value for the `best_params_` attribute
        mock_pipeline.best_params_ = {'param1': 1, 'param2': 2}

        mock_pipeline.named_steps = {'classifier': Mock()}

        # Set up the mock for Polynomial Features
        mock_polynomial_features_rv = Mock()
        mock_polynomial_features.return_value = mock_polynomial_features_rv

        # Set up the mock grid search object to return itself when the fit method is called
        mock_grid_search.return_value = mock_grid_search

        mock_grid_search_fit = Mock()
        mock_grid_search.fit.return_value = mock_grid_search_fit
        # Create mock objects for the X and y inputs
        mock_X = Mock()
        mock_y = Mock()

        # Call the function with the mock objects and some test input values
        result = svd_n_range_scores_for_pipe(mock_pipeline, [1, 2, 3], no_poly_k=100, poly_2_k=200,
                                             defaults={'foo': 'bar'}, cv=mock_cv, X=mock_X, y=mock_y)

        # Assert that the function returns the expected result
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 4)

        for v in result.values():
            self.assertIn('range', v)
            self.assertIn('no_svd', v)
            self.assertEqual(mock_grid_search_fit, v['range'])
            self.assertEqual(mock_grid_search_fit, v['no_svd'])

        # Assert that the mock grid search object and it's fit method were called 8 times with proper arguments
        self.assertEqual(mock_grid_search.call_count, 8)
        self.assertEqual(mock_grid_search.fit.call_count, 8)

        call = mock_grid_search.fit.call_args_list[0]
        assert all(x[1]['cv'] == mock_cv and x[1]['scoring'] == 'f1' for x in mock_grid_search.call_args_list)
        assert all(x[0] == (mock_X, mock_y) for x in mock_grid_search.fit.call_args_list)

    def test_mislabeled_dups(self):
        df = pd.DataFrame({'text': ['text1', 'text2', 'text1', 'text2'], 'target': [1, 0, 0, 0]})
        result = mislabeled_dups(df)
        print(result)
        expected_result = pd.Series(['text1', 'text1'], [0, 2], name='text')
        pd.testing.assert_series_equal(result, expected_result)

    def test_drop_mislabeled_dups(self):
        df = pd.DataFrame({'text': ['text1', 'text2', 'text1', 'text2'], 'target': [1, 0, 0, 0]})
        result = drop_mislabeled_dups(df)
        print(result)
        expected_result = pd.DataFrame({'text': ['text2', 'text2'], 'target': [0, 0]},
                                       index=pd.RangeIndex(start=1, stop=4, step=2))
        pd.testing.assert_frame_equal(result, expected_result)
