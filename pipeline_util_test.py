from unittest.mock import patch
import pandas as pd
from pipeline_util import *


import unittest.mock

class TestFunctions(unittest.TestCase):
    
    def test_series_to_dict_single_word_elements(self):
        # Test Series with single word elements
        s = pd.Series(['apple', 'banana', 'cherry'], name='column_name')
        result = series_to_dict(s)
        self.assertEqual(result, [{'column_name': 'apple'}, {'column_name': 'banana'}, {'column_name': 'cherry'}])
        
    def test_make_kw_pipeline(self):
        # Test that make_kw_pipeline creates a Pipeline with the proper transformers
        kw_pipeline = make_kw_pipeline()
        self.assertIsInstance(kw_pipeline, Pipeline)
        self.assertIsInstance(kw_pipeline[0][1], FunctionTransformer)
        self.assertEqual(kw_pipeline[0][1].func, series_to_dict)

    def test_make_kw_pipeline(self):
        # Test that the returned object is a Pipeline
        pipeline = make_kw_pipeline()
        self.assertIsInstance(pipeline, Pipeline)

        # Test that the Pipeline has the correct steps
        self.assertEqual(pipeline.steps[0][0], 'kw_to_dict')
        self.assertEqual(pipeline.steps[1][0], 'd_vect')

    def test_make_loc_pipeline(self):
        # Test that make_loc_pipeline creates a Pipeline with the proper transformers
        loc_pipeline = make_loc_pipeline()
        self.assertIsInstance(loc_pipeline, Pipeline)
        self.assertIsInstance(loc_pipeline[0][1], FunctionTransformer)
        self.assertEqual(loc_pipeline[0][1].func, series_to_dict)
        
    def test_make_poly2_k_best_pipeline(self):
        # Test that the returned object is a Pipeline
        pipeline = make_poly2_k_best_pipeline()
        self.assertIsInstance(pipeline, Pipeline)

        # Test that the Pipeline has the correct steps
        self.assertEqual(pipeline.steps[0][0], 'poly2')
        self.assertEqual(pipeline.steps[1][0], 'k_best')


    @patch('pipeline_util.make_kw_pipeline')
    def test_make_vectorizer_with_kw(self, mock_make_kw_pipeline):
        # Set up mock return value for make_kw_pipeline
        mock_make_kw_pipeline.return_value = 'mock_kw_pipeline'

        # Call make_vectorizer with kw=True
        vectorizer = make_vectorizer(kw=True)

        # Assert that make_kw_pipeline was called
        mock_make_kw_pipeline.assert_called_once()

        # Assert that the returned ColumnTransformer has the correct column
        self.assertEqual(vectorizer.transformers[1][0], 'kw_dict_vect')
        self.assertEqual(vectorizer.transformers[1][1], 'mock_kw_pipeline')
        self.assertEqual(vectorizer.transformers[1][2], 'keyword')

    @patch('pipeline_util.make_loc_pipeline')
    def test_make_vectorizer_with_loc(self, mock_make_loc_pipeline):
        # Set up mock return value for make_loc_pipeline
        mock_make_loc_pipeline.return_value = 'mock_loc_pipeline'

        # Call make_vectorizer with loc=True
        vectorizer = make_vectorizer(loc=True)

        # Assert that make_loc_pipeline was called
        mock_make_loc_pipeline.assert_called_once()

        # Assert that the returned ColumnTransformer has the correct column
        self.assertEqual(vectorizer.transformers[1][0], 'loc_dict_vect')
        self.assertEqual(vectorizer.transformers[1][1], 'mock_loc_pipeline')
        self.assertEqual(vectorizer.transformers[1][2], 'location')

    def test_make_vectorizer_with_hashtags(self):
        # Call make_vectorizer with hashtags=True
        vectorizer = make_vectorizer(hashtags=True)

        # Assert that the returned ColumnTransformer has the correct column
        self.assertEqual(vectorizer.transformers[1][0], 'hashtags_c_vect')
        self.assertIsInstance(vectorizer.transformers[1][1], CountVectorizer)
        self.assertEqual(vectorizer.transformers[1][2], 'hashtags')

    @patch('pipeline_util.make_preparation_pipeline')
    @patch('pipeline_util.make_poly2_k_best_pipeline')
    def test_make_transformation_pipeline(self, mock_make_poly2_k_best_pipeline, mock_make_preparation_pipeline):
        # Set up mock return value for make_preparation_pipeline
        mock_pipeline = mock_make_preparation_pipeline.return_value
        mock_pipeline.steps = ['step1', 'step2']

        # Set up mock return value for make_poly2_k_best_pipeline
        mock_poly2_k_best_pipeline = mock_make_poly2_k_best_pipeline.return_value
        mock_poly2_k_best_pipeline.steps = ['poly2_k_best_step1', 'poly2_k_best_step2']

        # Call make_transformation_pipeline and store the result in pipeline
        pipeline = make_transformation_pipeline(classifier='mock_classifier', params={'param1': 'value1'})

        # Assert that make_preparation_pipeline and make_poly2_k_best_pipeline were called with the correct arguments
        mock_make_preparation_pipeline.assert_called_with(param1='value1')
        mock_make_poly2_k_best_pipeline.assert_called_once()

        # Assert that pipeline.steps is as expected
        self.assertEqual(len(pipeline.steps), 5)
        self.assertEqual(pipeline.steps[0], 'step1')
        self.assertEqual(pipeline.steps[1], 'step2')
        self.assertEqual(pipeline.steps[2], ('poly2_k_best', mock_poly2_k_best_pipeline))
        self.assertIsInstance(pipeline.steps[3][1], TruncatedSVD)
        self.assertEqual(pipeline.steps[3][1].algorithm, 'arpack')
        self.assertEqual(pipeline.steps[4], ('classifier', 'mock_classifier'))

    def test_make_preparation_pipeline(self):
        # Test that the returned object is a Pipeline
        pipeline = make_preparation_pipeline()
        self.assertIsInstance(pipeline, Pipeline)

        # Test that the Pipeline has the correct steps
        self.assertEqual(pipeline.steps[0][0], 'url_cleaner')
        self.assertEqual(pipeline.steps[1][0], 'vectorizer')
        self.assertIsInstance(pipeline.steps[1][1], ColumnTransformer)