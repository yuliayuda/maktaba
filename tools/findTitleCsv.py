import pandas as pd
import math

def split_and_save_books(file_path, output_file, start_title=0, max_titles=None):
    # Membaca file CSV sumber
    print(f"Membaca file CSV: {file_path}")
    df = pd.read_csv(file_path)
    
    # Mengambil hanya kolom 'Book_name' dan menghapus duplikasi (jika ada)
    book_names = df['Book_name'].drop_duplicates().reset_index(drop=True)
    
    # Menghitung total judul buku
    total_books = len(book_names)
    
    # Menentukan batas pencatatan (defaultnya hingga akhir jika max_titles tidak diberikan)
    end_title = start_title + max_titles if max_titles else total_books
    
    # Pastikan batas pencatatan tidak melebihi jumlah judul yang tersedia
    if start_title >= total_books:
        print(f"Batas awal pencatatan melebihi total judul yang tersedia. Terdapat hanya {total_books} judul.")
        return
    
    end_title = min(end_title, total_books)  # Batas tidak boleh melebihi total judul
    
    # Memilih judul buku yang sesuai dengan batas awal dan akhir pencatatan
    selected_books = book_names[start_title:end_title].reset_index(drop=True)
    
    # Menentukan berapa banyak judul buku dalam satu kolom
    books_per_column = 1000
    
    # Menghitung berapa banyak kolom yang diperlukan
    num_columns = math.ceil(len(selected_books) / books_per_column)
    
    # Membuat dictionary untuk menyimpan data dalam format yang diinginkan
    book_columns = {}
    
    for col in range(num_columns):
        start_index = col * books_per_column
        end_index = start_index + books_per_column
        # Ambil 1000 buku untuk setiap kolom
        book_columns[f'title {col}'] = selected_books[start_index:end_index].reset_index(drop=True)
    
    # Menggabungkan semua kolom menjadi DataFrame baru
    df_output = pd.DataFrame(book_columns)
    
    # Menyimpan hasil ke file CSV baru
    print(f"Menyimpan data ke file: {output_file}")
    df_output.to_csv(output_file, index=False)

# Contoh penggunaan:
file_path = '/kaggle/input/arabic-library/my_csv.csv'  # Ganti dengan path file CSV sumber
output_file = '/kaggle/working/book_titles.csv'  # Ganti dengan path file CSV tujuan

# Memulai pencatatan dari judul ke-5001 dan hanya mencatat 5000 judul
split_and_save_books(file_path, output_file, start_title=5000, max_titles=10000)
