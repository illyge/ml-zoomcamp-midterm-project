import unittest
from unittest.mock import patch, Mock

from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from train_pipeline import pipeline, prepare_data, train


class TestTrainPipeline(unittest.TestCase):
    @patch('train_pipeline.make_preparation_pipeline')
    def test_pipeline(self, mock_make_preparation_pipeline):
        mock_pipeline = mock_make_preparation_pipeline.return_value
        mock_pipeline.steps = [('step1', 'obj1'), ('step2', 'obj2')]

        result = pipeline()
        self.assertIsInstance(result, Pipeline)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.steps[:2], [('step1', 'obj1'), ('step2', 'obj2')])
        self.assertEqual(result.steps[2][0], 'classifier')
        self.assertIsInstance(result.steps[2][1], ComplementNB)

    @patch('train_pipeline.drop_mislabeled_dups')
    def test_prepare_data(self, mock_drop_mislabeled_dups):
        mock_df = Mock()
        mock_df_without_na = mock_df.fillna.return_value
        mock_df_without_mislabeled = mock_drop_mislabeled_dups.return_value

        result = prepare_data(mock_df)

        self.assertEqual(mock_df.fillna.call_count, 1)
        self.assertEqual(mock_df.fillna.call_args[0][0], "")

        self.assertEqual(mock_drop_mislabeled_dups.call_count, 1)
        self.assertEqual(mock_drop_mislabeled_dups.call_args[0][0], mock_df_without_na)

        self.assertEqual(result, mock_df_without_mislabeled)

    @patch('train_pipeline.pd')
    @patch('train_pipeline.bentoml')
    @patch('train_pipeline.pipeline')
    @patch('train_pipeline.prepare_data')
    def test_train(self, mock_prepare_data, mock_pipeline, mock_bentoml, mock_pd):
        mock_data = mock_pd.read_csv.return_value
        mock_classifier = mock_pipeline.return_value
        mock_prepared = mock_prepare_data.return_value

        train()

        self.assertEqual(mock_pd.read_csv.call_count, 1, msg='Should read the data from csv once')

        self.assertEqual(mock_prepare_data.call_count, 1, msg='Should prepare the data once')
        self.assertEqual(mock_prepare_data.call_args[0][0], mock_data,
                         msg='Should prepare the data based on the csv read')

        self.assertEqual(mock_classifier.fit.call_count, 1, msg='Should fit the model once')
        self.assertEqual(mock_classifier.fit.call_args[0], (mock_prepared, mock_prepared.target),
                         msg='Should fit the model with the correct data')

        self.assertEqual(mock_bentoml.sklearn.save_model.call_count, 1, msg='Should save model once')
        self.assertEqual(mock_bentoml.sklearn.save_model.call_args[0][1], mock_classifier,
                         msg='Should save the trained classifier')
