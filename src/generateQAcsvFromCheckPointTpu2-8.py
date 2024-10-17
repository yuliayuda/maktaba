import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
from sklearn.model_selection import train_test_split
import wandb
from huggingface_hub import login
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Login ke W&B menggunakan API key
wandb.login(key="5a646a4e2997f8ff868dfe4df325accd8a13b4b6")
login(token="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")

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

# Fine-tuning menggunakan Trainer dari Hugging Face dengan checkpointing
def fine_tune_model(train_dataset, val_dataset, output_dir="/content/drive/Shareddrives/Gpldome_2/output_model", epochs=3, batch_size=4, resume_from_checkpoint=None):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_steps=500,  # Simpan checkpoint setiap 500 langkah
        eval_steps=500,
        logging_dir='./logs',
        logging_steps=100,
        num_train_epochs=epochs,
        save_total_limit=2,  # Simpan hanya 2 checkpoint terbaru
        load_best_model_at_end=True,
        # Resume from last checkpoint jika tersedia
        resume_from_checkpoint=resume_from_checkpoint  # Resume dari checkpoint terakhir
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Melanjutkan fine-tuning dari checkpoint terakhir jika diberikan
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# Simpan model yang sudah di fine-tune
def save_fine_tuned_model(model, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Fungsi utama untuk menjalankan fine-tuning
def main(csv_file, resume_from_checkpoint=None):
    # Load dan bagi dataset
    train_data, val_data = load_and_split_data(csv_file)

    # Siapkan dataset untuk fine-tuning
    train_dataset = ArabicQuestionAnswerDataset(train_data, tokenizer)
    val_dataset = ArabicQuestionAnswerDataset(val_data, tokenizer)

    # Fine-tune model, jika ada checkpoint, mulai dari checkpoint tersebut
    fine_tune_model(train_dataset, val_dataset, resume_from_checkpoint=resume_from_checkpoint)

    # Simpan model yang sudah di fine-tune
    save_fine_tuned_model(model, "/content/drive/Shareddrives/Gpldome_2/fine_tuned_arabic_question_model")

    # Hasilkan pertanyaan dan jawaban serta simpan ke CSV
    df = pd.read_csv(csv_file)
    df['Generated_Question'] = df['text'].apply(lambda x: generate_questions(model, tokenizer, x))
    df['Generated_Answer'] = df['text']
    
    # Simpan ke file CSV baru
    output_csv = "/content/drive/Shareddrives/Gpldome_2/generated_questions_answers.csv"
    df.to_csv(output_csv, index=False)
    print(f"Hasil pertanyaan dan jawaban disimpan di: {output_csv}")

if __name__ == "__main__":
    csv_file = "/content/books.csv"
    
    # Cek apakah ada checkpoint yang bisa dilanjutkan
    checkpoint_dir = "/content/drive/Shareddrives/Gpldome_2/output_model"
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        resume_from_checkpoint = checkpoint_dir  # Mulai dari checkpoint terakhir
    else:
        resume_from_checkpoint = None  # Jika tidak ada checkpoint, mulai dari awal
    
    main(csv_file, resume_from_checkpoint=resume_from_checkpoint)
