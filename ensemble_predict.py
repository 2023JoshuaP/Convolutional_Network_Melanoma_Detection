import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === CONFIGURACIÓN ===
MODELS_DIR = "skin_cancer_mnist/models"
IMAGE_DIR = "skin_cancer_mnist/data/raw/HAM10000_images_part_1"
METADATA_CSV = "skin_cancer_mnist/data/raw/HAM10000_metadata.csv"
OUTPUT_CSV = "ensemble_predictions.csv"
IMG_SIZE = (160, 160)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# === Cargar modelos ===
print("📦 Cargando modelos...")
models = []
for i in range(1, 6):
    path = os.path.join(MODELS_DIR, f"model_fold{i}.h5")
    print(f"  ↪️  {path}")
    models.append(tf.keras.models.load_model(path))

# === Cargar metadata (si existe) ===
metadata = pd.read_csv(METADATA_CSV)
metadata = metadata.set_index("image_id")

# === Procesar imágenes ===
resultados = []
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

print(f"\n🖼️ Procesando {len(image_files)} imágenes...")
for img_file in image_files:
    image_id = img_file.replace(".jpg", "")
    img_path = os.path.join(IMAGE_DIR, img_file)

    # Leer imagen y preprocesar
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predicción por cada modelo
    predicciones = []
    for model in models:
        pred = model.predict(img_batch, verbose=0)
        pred_class = CLASS_NAMES[np.argmax(pred[0])]
        predicciones.append(pred_class)

    # Voto por mayoría
    voto_final = Counter(predicciones).most_common(1)[0][0]

    # Buscar etiqueta real
    if image_id in metadata.index:
        etiqueta_real = metadata.loc[image_id]['dx']
    else:
        etiqueta_real = "?"

    resultados.append({
        "image_id": image_id,
        "real": etiqueta_real,
        "predicha_1": predicciones[0],
        "predicha_2": predicciones[1],
        "predicha_3": predicciones[2],
        "predicha_4": predicciones[3],
        "predicha_5": predicciones[4],
        "voto_final": voto_final
    })

# === Guardar resultados ===
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Resultados guardados en {OUTPUT_CSV}")
