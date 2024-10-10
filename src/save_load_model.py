import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_dir):
    """Memuat model dan tokenizer dari direktori yang ditentukan."""
    if not os.path.exists(model_dir):
        print(f"Direktori {model_dir} tidak ditemukan.")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    print(f"Model dan tokenizer dimuat dari {model_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    model_dir = './models/arabic_sentence_transformer'  # Ganti dengan path yang sesuai

    # Contoh pemuatan model
    model, tokenizer = load_model(model_dir)

    # Anda dapat melanjutkan dengan menggunakan model dan tokenizer ini untuk inferensi
