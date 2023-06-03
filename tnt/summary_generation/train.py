import csv
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from config import Config


def run_training():
    config = Config()

    # Load the training data from the CSV file
    training_data = load_data(config.DATA_CSV)

    # Preprocess the training data
    preprocessed_data = preprocess_data(training_data, config)

    # Train the summary generation model
    train_model(preprocessed_data, config)


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def preprocess_data(data, config):
    preprocessed_data = []
    tokenizer = T5Tokenizer.from_pretrained(config.SUMMARY_MODEL)

    for item in data:
        text = item['text']
        summary = item['summary']

        # Tokenize the input text and summary
        input_ids = tokenizer.encode(text, truncation=True, padding='max_length', max_length=config.MAX_SUMMARY_LENGTH)
        target_ids = tokenizer.encode(summary, truncation=True, padding='max_length', max_length=config.MAX_SUMMARY_LENGTH)

        # Add the preprocessed data to the list
        preprocessed_data.append((input_ids, target_ids))

    return preprocessed_data


def train_model(data, config):
    model = T5ForConditionalGeneration.from_pretrained(config.SUMMARY_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(config.SUMMARY_MODEL)

    # Set up the training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0
        for input_ids, target_ids in data:
            # Convert the input and target to tensors
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            target_ids = torch.tensor(target_ids).unsqueeze(0)

            # Clear the gradients
            model.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print the average loss for the epoch
        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss}")

    # Save the trained model
    model.save_pretrained(config.SAVED_MODEL_DIR)


if __name__ == '__main__':
    run_training()
