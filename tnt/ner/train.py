import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from config import Config

def preprocess_text(text):
    # Add any necessary preprocessing steps
    # ...

    return text

def perform_ner(test_data, tokenizer, model):
    # Initialize the results list
    results = []

    # Perform NER on each example in the test data
    for _, row in test_data.iterrows():
        text = row['text']

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Tokenize the text
        inputs = tokenizer.encode_plus(preprocessed_text, truncation=True, padding='max_length', max_length=Config.MAX_SECTION_LENGTH, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Perform NER using the model
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_labels = torch.argmax(outputs.logits, dim=2)[0]

        # Extract named entities and their labels
        entities = []
        current_entity = ""
        current_label = ""
        for token, label_id in zip(tokenizer.tokenize(preprocessed_text), predicted_labels):
            label = tokenizer.decode(label_id)
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = token
                current_label = label[2:]
            elif label.startswith('I-'):
                if current_entity:
                    current_entity += " " + token
            else:
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = ""
                current_label = ""
        
        if current_entity:
            entities.append((current_entity, current_label))

        # Add the results to the list
        results.append((text, entities))

    return results

def run_tests():
    # Load the test data from CSV
    test_data = pd.read_csv(Config.TEST_DATA_CSV)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(Config.NER_MODEL)

    # Perform NER on the test data
    results = perform_ner(test_data, tokenizer, model)

    # Print the results
    for text, entities in results:
        print(f"Text: {text}")
        print("Named Entities:")
        for entity, label in entities:
            print(f"- Entity: {entity}, Label: {label}")
        print()

if __name__ == '__main__':
    run_tests()
