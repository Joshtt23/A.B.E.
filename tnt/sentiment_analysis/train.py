import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer

from tnt.ner import preprocess_text, train_ner_model

def train_ner():
    # Load the training data from CSV
    train_data = pd.read_csv('tnt/data.csv')

    # Initialize the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

    # Perform the training loop
    for _, row in train_data.iterrows():
        text = row['text']
        labels = row['label_keyword']

        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Train the model on the preprocessed text and labels
        train_ner_model(model, tokenizer, preprocessed_text, labels)

    # Save the trained model
    model.save_pretrained('saved_model')

if __name__ == '__main__':
    train_ner()
