import tensorflow as tf
import os

# Dosya yolu
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "final_model_garanti.keras")

# 1. Modeli Yükleme
print("Model dosyası okunuyor...")
model = tf.keras.models.load_model(model_path)

# 2. Ozet
print("\nModelin Mimarisi:\n")
model.summary()

weights = model.layers[0].get_weights()[0]
print(f"\nİlk katman ağırlık boyutu: {weights.shape}")

