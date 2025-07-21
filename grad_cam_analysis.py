import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# === Configuraciones ===
IMG_SIZE = (160, 160)
FOLDER1 = "HAM10000_images_part_1"
FOLDER2 = "HAM10000_images_part_2"
CSV_PATH = "HAM10000_metadata.csv"
MODEL_PATH = "model_fold5.h5"
OUTPUT_DIR = "gradcam_outputs"
NUM_IMAGES = 50  # <-- Límite de imágenes por carpeta

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Cargar modelo y datos ===
model = load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
df["dx_idx"] = le.fit_transform(df["dx"])

# === Grad-CAM helper ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# === Procesamiento por lote ===
resultados = {
    "imagen": [],
    "real": [],
    "predicha": []
}

def procesar_imagenes_en(folder):
    archivos = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    seleccionados = random.sample(archivos, min(NUM_IMAGES, len(archivos)))

    for filename in seleccionados:
        file_id = filename.split('.')[0]
        row = df[df['image_id'] == file_id]
        if row.empty:
            continue

        real_label = row['dx'].values[0]
        resultados["imagen"].append(file_id)
        resultados["real"].append(real_label)

        # === Preparar imagen ===
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array_exp = np.expand_dims(img_array, axis=0)

        # === Predicción ===
        pred = model.predict(img_array_exp, verbose=0)
        pred_class = np.argmax(pred[0])
        pred_label = le.inverse_transform([pred_class])[0]
        resultados["predicha"].append(pred_label)

        # === Grad-CAM ===
        heatmap = make_gradcam_heatmap(img_array_exp, model)
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(np.uint8(img_array * 255), 0.6, heatmap_color, 0.4, 0)

        # === Guardar imagen con Grad-CAM ===
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}_gradcam.jpg")
        cv2.imwrite(output_path, superimposed_img)

# === Ejecutar en ambas carpetas ===
procesar_imagenes_en(FOLDER1)
procesar_imagenes_en(FOLDER2)

# === Guardar resultados ===
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("predicciones_comparadas.csv", index=False)

# === Graficar comparación ===
plt.figure(figsize=(10, 6))
sns.countplot(data=df_resultados.melt(id_vars="imagen", value_vars=["real", "predicha"]),
              x="value", hue="variable", palette="Set2")
plt.title("Comparación de etiquetas reales vs predichas (100 imágenes)")
plt.xlabel("Clases")
plt.ylabel("Cantidad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("comparacion_clases.png")
plt.close()
