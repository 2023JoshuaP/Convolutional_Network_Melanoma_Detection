import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# === Cargar CSV ===
df = pd.read_csv("ensemble_predictions.csv")
folds = ['predicha_1', 'predicha_2', 'predicha_3', 'predicha_4', 'predicha_5']
class_labels = sorted(df['real'].unique())

# === Calcular precisión por clase y fold ===
results = []
for fold in folds:
    for label in class_labels:
        subset = df[df['real'] == label]
        acc = accuracy_score(subset['real'], subset[fold])
        results.append({"fold": fold, "class": label, "accuracy": round(acc * 100, 2)})

# === Mostrar tabla de precisión por clase y modelo ===
df_results = pd.DataFrame(results)
print(df_results.pivot(index='class', columns='fold', values='accuracy'))

# === Graficar con etiquetas encima de cada barra ===
sns.set(style="whitegrid")
g = sns.catplot(
    data=df_results,
    kind="bar",
    x="class",
    y="accuracy",
    hue="fold",
    height=6,
    aspect=2,
    palette="viridis"
)

# Ajustar ejes y título
g.set_axis_labels("Clase (dx)", "Precisión (%)")
g.set_titles("Precisión por clase y modelo")
g.set(ylim=(0, 100))

# Añadir etiquetas encima de cada barra
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=2)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
