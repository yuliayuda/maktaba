import pandas as pd

# Muat dataset dari file CSV
df = pd.read_csv('/kaggle/input/arabic-library/my_csv.csv')

# Muat daftar judul buku dari file teks
with open('/kaggle/input/bookst2/book_titles1.txt', 'r', encoding='utf-8') as file:
    book_titles = [line.strip() for line in file.readlines()]

# Inisialisasi daftar untuk menyimpan judul buku yang tidak ditemukan
not_found_titles = []

# Inisialisasi DataFrame kosong untuk hasil
filtered_df = pd.DataFrame()

# Mencari setiap judul buku
for title in book_titles:
    # Filter untuk mengambil semua baris yang memiliki judul buku
    matching_rows = df[df['Book_name'].str.contains(title, case=False, na=False)]
    
    # Jika ada baris yang cocok, tambahkan ke filtered_df
    if not matching_rows.empty:
        filtered_df = pd.concat([filtered_df, matching_rows], ignore_index=True)
    else:
        # Jika tidak ditemukan, tambahkan judul ke not_found_titles
        not_found_titles.append(title)

# Simpan hasil yang ditemukan ke file CSV
output_file = '/kaggle/working/books.csv'
filtered_df.to_csv(output_file, index=False)

# Simpan judul buku yang tidak ditemukan ke file log
if not_found_titles:
    with open('/kaggle/working/not_found.log', 'w', encoding='utf-8') as log_file:
        for title in not_found_titles:
            log_file.write(f"{title}\n")

print(f"Hasil telah disimpan dalam {output_file}.")
if not_found_titles:
    print(f"Judul buku yang tidak ditemukan telah dicatat dalam not_found_titles.log.")
