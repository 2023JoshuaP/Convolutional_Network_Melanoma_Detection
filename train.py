import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from model_utils import create_model

# === Cargar datos preprocesados ===
X = np.load("data_processed/images_160.npy")
y = np.load("data_processed/labels_160.npy")
class_names = np.load("data_processed/class_names.npy")

# === Hiperparámetros ===
num_folds = 5
batch_size = 32
epochs = 50
input_shape = (160, 160, 3)
num_classes = len(np.unique(y))

# === Validación cruzada estratificada ===
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold = 1

for train_index, val_index in kf.split(X, y):
    print(f"\n📦 Entrenando Fold {fold}...\n")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # One-hot encoding
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

    # Crear modelo
    model = create_model(input_shape=input_shape, num_classes=num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    # Entrenamiento
    history = model.fit(X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stop],
                        verbose=1)

    # Evaluación
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Matriz de confusión
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matriz de Confusión - Fold {fold}")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_fold{fold}.png")
    plt.close()

    # Guardar modelo
    model.save(f"model_fold{fold}.h5")
    print(f"💾 Modelo guardado como model_fold{fold}.h5")

    fold += 1
