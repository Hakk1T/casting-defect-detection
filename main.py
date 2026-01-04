import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Dosya yolu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "casting_data")
TRAIN_DIR = os.path.join(IMAGE_DIR, "train")
TEST_DIR = os.path.join(IMAGE_DIR, "test")

print(f"Veriler şuradan okunacak: {IMAGE_DIR}")

# On isleme
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32  
SEED_NUMBER = 123

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.75, 1.25],
    rescale=1./255,
    validation_split=0.2
)

# Test verisi için sadece ölçekleme yapılır
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Veriyi Yükleme
print("Eğitim verisi yükleniyor...")
train_dataset = train_generator.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes={"ok_front": 0, "def_front": 1},
    shuffle=True,
    seed=SEED_NUMBER,
    subset="training"
)

print("Doğrulama (Validation) verisi yükleniyor...")
validation_dataset = train_generator.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes={"ok_front": 0, "def_front": 1},
    shuffle=True,
    seed=SEED_NUMBER,
    subset="validation"
)

print("Test verisi yükleniyor...")
test_dataset = test_generator.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes={"ok_front": 0, "def_front": 1},
    shuffle=False,
    seed=SEED_NUMBER
)

# Veri seti için analiz tablosu
try:
    image_data = []
    for dataset, typ in zip([train_dataset, validation_dataset, test_dataset], ["train", "validation", "test"]):
        for name in dataset.filenames:
            image_data.append({
                "data": typ,
                "class": name.split(os.path.sep)[-2], 
                "filename": name.split(os.path.sep)[-1]
            })

    image_df = pd.DataFrame(image_data)
    
    # Tablo
    if not image_df.empty:
        data_crosstab = pd.crosstab(index=image_df["data"],
                                    columns=image_df["class"],
                                    margins=True,
                                    margins_name="Total")
        print("\nVeri Dağılımı Tablosu:")
        print(data_crosstab)
    else:
        print("Uyarı: DataFrame oluşturulamadı çünkü veri bulunamadı.")

except Exception as e:
    print(f"Veri analizi kısmında önemsiz bir hata oluştu, devam ediliyor: {e}")

model = tf.keras.models.Sequential([
    # 1. Konvolüsyon Bloğu (Giris)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SIZE + (1,)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 2. Konvolüsyon Bloğu
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 3. Konvolüsyon Bloğu
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Düzleştirme ve Sınıflandırma 
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Aşırı öğrenmeyi engellemek en uygun deger 0.5 
    
    # Çıkış Katmanı 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modeli Derleme 
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Özeti
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "cnn_casting_model.keras", 
    verbose=1, 
    save_best_only=True, 
    monitor="val_loss"
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs= 3, 
    callbacks=[checkpoint]
)

#Sonuc kismi
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Eğitim ve Doğrulama Başarısı')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

save_path = os.path.join(BASE_DIR, "final_model_garanti.keras")
model.save(save_path)

print("-" * 30)
print(f"DOSYA OLUŞTURULDU: {save_path}")
print("-" * 30)

# Sonuclari cizdirme
plt.figure(figsize=(10, 6))
print("Proje başarıyla tamamlandı!")
