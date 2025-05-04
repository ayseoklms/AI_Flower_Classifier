import os
import shutil
import random
import pathlib 

# Orijinal indirdiğiniz 'flowers' klasörünün yolu
SOURCE_DIR = r"C:\Users\ayseo\OneDrive\Masaüstü\flowers"
DEST_DIR = r"C:\Users\ayseo\OneDrive\Masaüstü\ai_project\data" 

# Eğitim seti için ayrılacak veri oranı (%80)
split_ratio = 0.8
train_path = pathlib.Path(DEST_DIR) / "train"
validation_path = pathlib.Path(DEST_DIR) / "validation"

print(f"Kaynak Dizin: {SOURCE_DIR}")
print(f"Hedef Dizin: {DEST_DIR}")
print(f"Eğitim yolu: {train_path}")
print(f"Doğrulama yolu: {validation_path}")
print("-" * 30)

# Kaynak dizin var mı kontrol et
if not os.path.exists(SOURCE_DIR):
    print(f"HATA: Kaynak dizin bulunamadı: {SOURCE_DIR}")
    exit() 

pathlib.Path(DEST_DIR).mkdir(parents=True, exist_ok=True)

try:

    class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    if not class_names:
        print(f"HATA: Kaynak dizin ({SOURCE_DIR}) içinde sınıf alt klasörleri bulunamadı.")
        exit()

    print(f"Bulunan sınıflar: {class_names}")
    print("-" * 30)

    for class_name in class_names:
        print(f"İşlenen sınıf: {class_name}")
        source_class_path = pathlib.Path(SOURCE_DIR) / class_name
        dest_train_class_path = train_path / class_name
        dest_validation_class_path = validation_path / class_name

        dest_train_class_path.mkdir(parents=True, exist_ok=True)
        dest_validation_class_path.mkdir(parents=True, exist_ok=True)

        all_files = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]

       
        image_files = all_files

        if not image_files:
            print(f"  -> Uyarı: {class_name} klasöründe hiç dosya bulunamadı.")
            continue

        # Dosyaları rastgele karıştırmak için
        random.shuffle(image_files)

        # Eğitim ve doğrulama setleri için dosya sayılarını hesapla
        num_files = len(image_files)
        num_train = int(num_files * split_ratio)
        num_validation = num_files - num_train

        print(f"  -> Toplam {num_files} dosya bulundu.")
        print(f"  -> Eğitim setine {num_train} dosya ayrılacak.")
        print(f"  -> Doğrulama setine {num_validation} dosya ayrılacak.")

        # Eğitim dosyalarını kopyala
        train_files = image_files[:num_train]
        for file_name in train_files:
            source_file = source_class_path / file_name
            dest_file = dest_train_class_path / file_name
            shutil.copy2(source_file, dest_file) 

        # Doğrulama dosyalarını kopyala
        validation_files = image_files[num_train:]
        for file_name in validation_files:
            source_file = source_class_path / file_name
            dest_file = dest_validation_class_path / file_name
            shutil.copy2(source_file, dest_file)

        print(f"  -> {class_name} sınıfı için kopyalama tamamlandı.")
        print("-" * 10)

    print("=" * 30)
    print("Veri ayırma işlemi başarıyla tamamlandı!")
    print(f"Eğitim verileri şurada: {train_path}")
    print(f"Doğrulama verileri şurada: {validation_path}")
    print("=" * 30)

except Exception as e:
    print(f"\n!!! Bir Hata Oluştu !!!")
    print(e)
    print("Lütfen dosya yollarını ve izinleri kontrol edin.")