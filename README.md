# Endüstriyel Döküm Hataları Tespiti (CNN Projesi)

Bu proje, döküm ürünlerinin (casting products) üretim hattındaki görsel kalite kontrolünü yapay zeka ile analizini yapar.

# Veri Seti
Kullanılan veri seti: **Casting Product Image Data for Quality Inspection**
Kullanılan veri seti dosyanın içerisinde **casting_data/** klasöründe bulunmaktadır.

Bu repo, dosya boyutu sınırları nedeniyle veri setini ve eğitilmiş `.keras` model dosyasını içermez. Projeyi çalıştırmak için:

1. **Veri Setini İndirin:**
   [Kaggle: Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) adresinden veri setini indirin.
2. **Klasöre Çıkartın:**
   İndirdiğiniz `casting_data` klasörünü projenin ana dizinine atın.
3. **Modeli Eğitin:**
   Aşağıdaki komutu çalıştırarak modeli eğitin ve `.keras` dosyasını oluşturun:
   ```bash
   python train_model.py

# Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza indirin :
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
   cd REPO_ADIN

2. Gerekli kütüphaneleri**requirements.txt** yükleyin :    
    
    ```bash
    pip install -r requirements.txt 

3. Modeli eğitin ve test edin :    
    
    ```bash
    python main.py

# Sonuçlar 

Model eğitim sonunda **final_model_garanti.keras** adlı dosyayı oluşturur. 
