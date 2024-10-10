import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

# Memuat dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df['text'].tolist(), df['Book_Number'].tolist()  # Ganti dengan kolom label yang sesuai

def train_model(train_file, output_dir='./models/arabic_sentence_transformer'):
    # Memuat dataset
    texts, labels = load_dataset(train_file)

    # Memuat tokenizer dan model
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/arabert")
    model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/arabert", num_labels=len(set(labels)))

    # Tokenisasi data
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Mengatur argumen pelatihan
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Mengatur Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokens,  # Sesuaikan jika Anda menggunakan dataset yang lebih kompleks
    )

    # Melatih model
    trainer.train()

    # Menyimpan model dan tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model dan tokenizer disimpan di {output_dir}")

if __name__ == "__main__":
    train_file = 'data/preprocessed_data.csv'  # Ganti dengan path yang sesuai
    train_model(train_file)
