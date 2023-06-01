import unittest
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer

from tnt.ner import preprocess_text, extract_entities, evaluate_entities

class NERTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the trained model and tokenizer
        cls.model = AutoModelForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
        cls.tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

    def test_extract_entities(self):
        # Test a specific example
        text = "Apple Inc. is a technology company based in California."
        expected_entities = ["Apple Inc.", "California"]
        preprocessed_text = preprocess_text(text)
        entities = extract_entities(self.model, self.tokenizer, preprocessed_text)
        self.assertEqual(entities, expected_entities)

    def test_evaluate_entities(self):
        # Load the test data from CSV
        test_data = pd.read_csv('tnt/data.csv')

        # Extract the required columns from the CSV data
        texts = test_data['text']
        expected_entities = test_data['label_entity']

        # Evaluate the model on the test data
        accuracy = evaluate_entities(texts, expected_entities, self.model, self.tokenizer)

        # Assert that accuracy is within a reasonable range
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
