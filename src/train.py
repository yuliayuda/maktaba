import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

def train_model(train_file, model_name='aubmindlab/arabert', output_dir='./models/arabic_sentence_transformer'):
    # Memuat dataset
    df = pd.read_csv(train_file)

    # Memeriksa apakah dataset kosong
    if df.empty:
        print("Dataset kosong!")
        return

    # Menggunakan tokenizer dari model yang dipilih
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenisasi data
    tokens = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # Membuat label (di sini diasumsikan ada kolom label, jika tidak, sesuaikan sesuai kebutuhan)
    labels = df['Book_Number'].tolist()  # Ganti dengan kolom label yang sesuai

    # Membagi data menjadi dataset dan label
    dataset = torch.utils.data.TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(labels))

    # Memuat model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))

    # Mengatur parameter pelatihan
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Menginisialisasi Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Melatih model
    trainer.train()

    # Menyimpan model yang sudah dilatih
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_file = 'data/preprocessed_data.csv'  # Ganti dengan path yang sesuai
    train_model(train_file)
