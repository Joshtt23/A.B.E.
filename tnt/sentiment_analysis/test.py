import unittest
from transformers import pipeline
import csv
from config import Config


class SummaryGenerationTestCase(unittest.TestCase):

    def load_data_from_csv(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data

    def test_summary_generation(self):
        config = Config()

        # Load the test data
        data = self.load_data_from_csv(config.DATA_CSV)

        # Load the trained model
        summarizer = pipeline(
            "summarization",
            model=config.SAVED_MODEL_DIR,
            tokenizer=config.SUMMARY_MODEL,
            framework="pt",
        )

        # Evaluate the model on the test data
        for example in data:
            text = example["text"]
            true_summary = example["summary"]

            # Generate the summary
            result = summarizer(
                text,
                max_length=config.MAX_SUMMARY_LENGTH,
                min_length=config.MIN_LENGTH,
                do_sample=False,
            )
            generated_summary = result[0]["summary_text"]

            # Compare the true summary and the generated summary
            self.assertEqual(generated_summary, true_summary)


def run_tests():
    unittest.main()


if __name__ == "__main__":
    run_tests()
