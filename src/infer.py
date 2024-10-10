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
    # Contoh teks yang ingin diuji
    test_text = "الإهداء إلى والديّ الحبيبين رحمهما الله تعالى."
    prediction = infer(test_text)

    print(f"Prediksi untuk teks: '{test_text}' adalah label: {prediction}")
