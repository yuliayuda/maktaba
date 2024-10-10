import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def infer(text, model_dir='./models/arabic_sentence_transformer', model_name='aubmindlab/arabert'):
    # Memuat tokenizer dan model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Tokenisasi input teks
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Melakukan prediksi
    with torch.no_grad():
        outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
        predictions = torch.argmax(outputs.logits, dim=1)

    return predictions.item()

if __name__ == "__main__":
    # Peta label untuk mengganti angka dengan nama buku
    label_map = {
        0: "Buku 1",
        1: "Buku 2",
        # Tambahkan sesuai dengan label yang ada
    }

    # Contoh teks yang ingin diuji
    test_text = "الإهداء إلى والديّ الحبيبين رحمهما الله تعالى."
    prediction = infer(test_text)

    # Menampilkan hasil prediksi dengan label yang lebih deskriptif
    print(f"Prediksi untuk teks: '{test_text}' adalah label: {label_map.get(prediction, 'Label tidak dikenali')}")
