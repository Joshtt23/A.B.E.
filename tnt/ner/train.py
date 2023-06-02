import csv
from transformers import (
    pipeline,
    TrainingArguments,
    Trainer,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from config import Config


def run_training():
    config = Config()

    # Load the training data
    data = load_data_from_csv(config.DATA_CSV)

    # Prepare the data for training
    tokenizer = RobertaTokenizer.from_pretrained(config.NER_MODEL)
    labels = [label for label in config.SENTIMENT_LABELS]
    label_map = {label: i for i, label in enumerate(labels)}

    # Prepare the training arguments
    training_args = TrainingArguments(
        output_dir=config.SAVED_MODEL_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        save_total_limit=config.SAVE_LIMIT,
        learning_rate=config.LEARNING_RATE,
        logging_dir=config.LOG_DIR,
        logging_steps=config.LOG_STEPS,
        save_strategy=config.SAVE_STRATEGY,
    )

    # Prepare the model
    model = RobertaForSequenceClassification.from_pretrained(
        config.NER_MODEL, num_labels=len(labels)
    )

    # Prepare the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        tokenizer=tokenizer,
        data_collator=lambda data: tokenizer(
            data["text"], truncation=True, padding=True
        ),
        compute_metrics=None,
    )

    # Start the training
    trainer.train()

    # Save the trained model
    trainer.save_model(config.SAVED_MODEL_DIR)


def load_data_from_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


if __name__ == "__main__":
    run_training()
