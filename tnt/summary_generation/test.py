import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import Config

def preprocess_text(text):
    # Add any necessary preprocessing steps
    # ...

    return text

def generate_summary(model, tokenizer, text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=Config.MAX_LENGTH, truncation=True)
    outputs = model.generate(inputs, max_length=Config.MAX_SUMMARY_LENGTH, min_length=Config.MIN_LENGTH, length_penalty=Config.TARGET_LENGTH_RATIO, num_beams=Config.NUM_BEAMS, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

def evaluate_summary(test_data, model, tokenizer):
    # Initialize evaluation metrics
    total_examples = len(test_data)
    correct_predictions = 0

    # Evaluate each example in the test data
    for _, row in test_data.iterrows():
        text = row['text']
        reference_summary = row['summary']

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Generate the summary
        generated_summary = generate_summary(model, tokenizer, preprocessed_text)

        # Compare the generated summary with the reference summary
        if generated_summary.strip() == reference_summary.strip():
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_examples

    return accuracy

def run_tests():
    # Load the trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(Config.SUMMARY_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(Config.SUMMARY_MODEL)

    # Read the test data from CSV
    test_data = pd.read_csv(Config.TEST_DATA_CSV)

    # Extract the required columns from the CSV data
    texts = test_data['text']
    reference_summaries = test_data['summary']

    # Create a new DataFrame to hold the extracted columns
    extracted_data = pd.DataFrame({'text': texts, 'summary': reference_summaries})

    # Evaluate the model on the extracted data
    accuracy = evaluate_summary(extracted_data, model, tokenizer)

    # Print the test results
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    run_tests()
