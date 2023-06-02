import unittest
from transformers import pipeline
import csv
from config import Config

class NerModuleTestCase(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.ner_model = pipeline("ner", model=self.config.NER_MODEL, device=self.config.NER_DEVICE)

    def load_data_from_csv(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data

    def test_ner_module(self):
        data = self.load_data_from_csv(self.config.DATA_CSV)
        for row in data:
            text = row["text"]
            expected_entities = row["label_keyword"]

            # Perform NER prediction
            result = self.ner_model(text)

            # Extract entity labels from the prediction result
            predicted_entities = [entity["entity"] for entity in result]

            self.assertEqual(predicted_entities, expected_entities)


def run_tests():
    unittest.main()


if __name__ == '__main__':
    run_tests()
