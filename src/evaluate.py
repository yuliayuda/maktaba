import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report

def evaluate_model(test_file, model_name='aubmindlab/arabert', model_dir='./models/arabic_sentence_transformer'):
    # Memuat dataset
    df = pd.read_csv(test_file)

    # Menggunakan tokenizer dari model yang dipilih
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Tokenisasi data
    tokens = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # Membuat label (di sini diasumsikan ada kolom label, jika tidak, sesuaikan sesuai kebutuhan)
    labels = df['Book_Number'].tolist()  # Ganti dengan kolom label yang sesuai

    # Memuat model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Melakukan prediksi
    with torch.no_grad():
        outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
        predictions = torch.argmax(outputs.logits, dim=1).tolist()

    # Menampilkan laporan klasifikasi
    print(classification_report(labels, predictions))

if __name__ == "__main__":
    test_file = 'data/preprocessed_data.csv'  # Ganti dengan path yang sesuai
    evaluate_model(test_file)
