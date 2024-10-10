import pandas as pd
import re

def clean_text(text):
    # Menghapus karakter khusus dan angka
    text = re.sub(r'[^ء-ي\s]', '', text)
    # Menghapus spasi tambahan
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_file, output_file):
    # Membaca dataset
    df = pd.read_csv(input_file)

    # Menampilkan informasi awal
    print("Data sebelum pembersihan:")
    print(df.head())

    # Membersihkan kolom 'text'
    df['text'] = df['text'].apply(clean_text)

    # Menampilkan informasi setelah pembersihan
    print("Data setelah pembersihan:")
    print(df.head())

    # Menyimpan data yang sudah diproses
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'data/dataset.csv'          # Ganti dengan path yang sesuai
    output_file = 'data/preprocessed_data.csv'  # Ganti dengan path yang sesuai

    preprocess_data(input_file, output_file)
