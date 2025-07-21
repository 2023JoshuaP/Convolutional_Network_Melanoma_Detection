import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse

IMG_SIZE = 160  # Tamaño fijo

def find_image_file(image_id, image_dirs):
    for dir in image_dirs:
        file_path = os.path.join(dir, image_id + '.jpg')
        if os.path.exists(file_path):
            return file_path
    return None

def load_and_preprocess_images(csv_path, image_dirs):
    print("[INFO] Cargando metadatos...")
    metadata = pd.read_csv(csv_path)
    images = []
    labels = []

    print("[INFO] Procesando imágenes...")
    for _, row in metadata.iterrows():
        img_path = find_image_file(row['image_id'], image_dirs)
        if img_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float16) / 255.0
            images.append(img)
            labels.append(row['dx'])
        else:
            print(f"[WARN] Imagen no encontrada: {row['image_id']}.jpg")

    print(f"[INFO] Total de imágenes procesadas: {len(images)}")
    return np.array(images, dtype=np.float16), np.array(labels)

def encode_labels(labels):
    print("[INFO] Codificando etiquetas...")
    encoder = LabelEncoder()
    labels_enc = encoder.fit_transform(labels)
    return labels_enc, encoder.classes_

def save_data(images, labels, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'images_160.npy'), images)
    np.save(os.path.join(output_dir, 'labels_160.npy'), labels)
    np.save(os.path.join(output_dir, 'class_names.npy'), class_names)
    print(f"[INFO] Datos guardados en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesamiento HAM10000 (160x160)")
    parser.add_argument("--csv", required=True, help="Ruta al archivo HAM10000_metadata.csv")
    parser.add_argument("--images", nargs='+', required=True, help="Una o más rutas a carpetas de imágenes")
    parser.add_argument("--output", default="preprocessed_data", help="Directorio de salida")

    args = parser.parse_args()

    imgs, lbls = load_and_preprocess_images(args.csv, args.images)
    lbls_enc, class_names = encode_labels(lbls)
    save_data(imgs, lbls_enc, class_names, args.output)
