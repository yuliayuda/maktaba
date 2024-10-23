import pandas as pd
import os

def search_and_copy_book(file_path, search_title, start_percent=0, end_percent=100):
    # Membersihkan kelebihan spasi pada search_title
    search_title = search_title.strip()
    
    # Membaca file CSV sumber dengan indikator pembacaan
    print(f"Membaca: {file_path}")
    print(f"Mencari: {search_title}")
    print(f"Mulai: {start_percent}%")
    print(f"Sampai: {end_percent}%")
    df = pd.read_csv(file_path)
    
    total_rows = len(df)
    
    # Menghitung baris awal dan akhir berdasarkan persentase
    start_row = int((start_percent / 100) * total_rows)
    end_row = int((end_percent / 100) * total_rows)
    
    if start_row >= end_row or start_row >= total_rows:
        print("Persentase tidak valid.")
        return
    
    print(f"Pencarian akan dimulai dari baris ke-{start_row} hingga baris ke-{end_row} dari {total_rows} total baris.")
    
    # Memotong data sesuai dengan persentase yang diberikan
    df_segment = df.iloc[start_row:end_row]
    
    # Membersihkan spasi dari semua entri di kolom "Book_name"
    df_segment['Book_name'] = df_segment['Book_name'].str.strip()
    
    # Mencari judul buku di kolom "Book_name"
    df_filtered = df_segment[df_segment['Book_name'] == search_title]
    
    if df_filtered.empty:
        print(f"Buku dengan judul '{search_title}' tidak ditemukan di rentang baris yang dipilih.")
        return
    
    # Menghitung total halaman berdasarkan sum dari Paragraph_No
    total_pages = df_filtered['Paragraph_No'].sum()
    print(f"Book_name '{search_title}' ditemukan dengan total halaman: {total_pages}")
    
    # Menyalin data yang ditemukan ke file CSV baru
    output_name = f"{search_title}.csv"
    output_file = "/kaggle/working/{output_name}"
    # Menambahkan indikator progres pengkopian
    filtered_rows = len(df_filtered)
    progress_step = max(1, filtered_rows // 10)  # Update progres setiap 10% data
    
    print(f"Mulai menyalin data ke file: {output_file}")
    
    with open(output_file, mode='w', encoding='utf-8', newline='') as output:
        df_filtered.to_csv(output, index=False)
        
        for i, row in df_filtered.iterrows():
            if i % progress_step == 0:
                print(f"Progres: {int((i / filtered_rows) * 100)}%")
        
    print(f"Proses selesai! Data telah disalin ke {output_file}")

# Contoh penggunaan:
file_path = '/kaggle/input/arabic-library/my_csv.csv'  # Ganti dengan path file CSV sumber
search_title = 'انيس الفقهاء في تعريفات الألفاظ المتداولة بين الفقهاء'  # Ganti dengan judul buku yang ingin dicari

# Memulai pencarian dari 0% hingga 100% dari total baris
search_and_copy_book(file_path, search_title, start_percent=0, end_percent=100)
