#pip install pandas torch transformers evaluate scikit-learn wandb huggingface-hub transformers torch accelerate

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
from sklearn.model_selection import train_test_split
import wandb

# Login ke W&B menggunakan API key
wandb.login(key="5a646a4e2997f8ff868dfe4df325accd8a13b4b6")

# Inisialisasi Tokenizer dan Model
model_name = "MIIB-NLP/Arabic-question-generation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class ArabicQuestionAnswerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = self.data.loc[idx, 'text']
        answer = context  # Menggunakan context yang sama untuk answer

        # Tokenisasi
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        labels = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        ).input_ids

        # Mengembalikan sebagai dictionary yang sesuai
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


# Memuat dataset dari CSV dan bagi menjadi train-test split
def load_and_split_data(csv_file):
    df = pd.read_csv(csv_file)
    train_data, val_data = train_test_split(df, test_size=0.2)
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True)

# Menghasilkan pertanyaan menggunakan model yang di fine-tune
def generate_questions(model, tokenizer, context, max_length=50):
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
    outputs = model.generate(inputs, max_length=max_length)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Fine-tuning menggunakan Trainer dari Hugging Face
def fine_tune_model(train_dataset, val_dataset, output_dir="output_model", epochs=3, batch_size=4):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_dir='./logs',
        logging_steps=100,
        num_train_epochs=epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

# Simpan model yang sudah di fine-tune
def save_fine_tuned_model(model, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Fungsi utama untuk menjalankan fine-tuning
def main(csv_file):
    # Load dan bagi dataset
    train_data, val_data = load_and_split_data(csv_file)

    # Siapkan dataset untuk fine-tuning
    train_dataset = ArabicQuestionAnswerDataset(train_data, tokenizer)
    val_dataset = ArabicQuestionAnswerDataset(val_data, tokenizer)

    # Fine-tune model
    fine_tune_model(train_dataset, val_dataset)

    # Simpan model yang sudah di fine-tune
    save_fine_tuned_model(model, "fine_tuned_arabic_question_model")

    # Hasilkan pertanyaan dan jawaban serta simpan ke CSV
    df = pd.read_csv(csv_file)
    df['Generated_Question'] = df['text'].apply(lambda x: generate_questions(model, tokenizer, x))
    df['Generated_Answer'] = df['text']  # Answer tetap dari kolom text yang sama
    
    # Simpan ke file CSV baru
    output_csv = "generated_questions_answers.csv"
    df.to_csv(output_csv, index=False)
    print(f"Hasil pertanyaan dan jawaban disimpan di: {output_csv}")

if __name__ == "__main__":
    # Ganti dengan path dataset CSV Anda
    csv_file = "/kaggle/working/books.csv"
    main(csv_file)
