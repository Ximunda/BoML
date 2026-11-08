import pandas as pd
import numpy as np
import zipfile, io, urllib.request
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg') # Устанавливаем фоновый режим, чтобы программа не зависала на графиках
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

print("1. Загрузка данных...")
url = "https://archive.ics.uci.edu/static/public/577/codon+usage.zip"
with urllib.request.urlopen(url) as resp:
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    df = pd.read_csv(z.open(csv_name), low_memory=False)

print("Размер датасета:", df.shape)

print("\n2. Очистка и предобработка...")
df = df.dropna(subset=["Kingdom"]).reset_index(drop=True)
y = df["Kingdom"]

feature_cols = [c for c in df.columns if c not in ["Kingdom", "SpeciesName", "SpeciesID"]]
cat_cols = ["DNAtype"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Лечим текстовые значения в числовых колонках (та самая ошибка с 'non-B hepatitis virus')
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(df)
print("Размерность после обработки:", X_processed.shape)

print("\n3. Метод главных компонент (PCA)...")
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_processed)
eigenvalues = pca_full.explained_variance_

eig_table = pd.DataFrame({
    "Компонента": [f"PC{i+1}" for i in range(len(eigenvalues))],
    "Собственное значение": eigenvalues,
    "Доля дисперсии (%)": pca_full.explained_variance_ratio_ * 100,
    "Накопленная дисперсия (%)": np.cumsum(pca_full.explained_variance_ratio_) * 100
})
print("\n--- Таблица собственных значений (первые 10 компонент) ---")
print(eig_table.head(10).round(4))

feature_names = preprocessor.get_feature_names_out()
loadings_table = pd.DataFrame(pca_full.components_[:3].T, index=feature_names, columns=["PC1", "PC2", "PC3"])
top_features_pc1 = loadings_table.reindex(loadings_table['PC1'].abs().sort_values(ascending=False).index)

print("\n--- Топ-15 признаков с наибольшим вкладом в PC1 ---")
print(top_features_pc1.head(15).round(4))

print("\n4. Построение и сохранение графиков (подожди пару секунд)...")

# 2D График
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_processed)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca2[:, 0], y=X_pca2[:, 1], hue=y, palette='tab20', s=20, alpha=0.7)
plt.title('2D Карта датасета (PCA)')
plt.xlabel('PC1 (29.23%)')
plt.ylabel('PC2 (18.56%)')
plt.savefig('plot_2d.png', dpi=300)
plt.close()

# 3D График
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_processed)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
kingdoms = y.unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(kingdoms)))
for kingdom, color in zip(kingdoms, colors):
    mask = (y == kingdom).values
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], label=kingdom, color=color, s=15, alpha=0.6)
ax.set_title('3D Карта датасета (PCA)')
ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', title="Kingdom")
plt.savefig('plot_3d.png', dpi=300) # Убрали bbox_inches
plt.close()

# График дисперсии
# График дисперсии
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', markersize=3)
axes[0].axhline(y=1.0, color='r', linestyle='--', label='Правило Кайзера')
axes[0].set_title('Scree Plot (Собственные значения)')
axes[0].legend()
axes[1].plot(range(1, len(eigenvalues)+1), np.cumsum(pca_full.explained_variance_ratio_) * 100, 'o-', markersize=3, color='g')
axes[1].axhline(y=90, color='r', linestyle='--', label='90% дисперсии')
axes[1].set_title('Кумулятивная дисперсия')
axes[1].legend()

try:
    plt.savefig('plot_variance.png', dpi=150)
    print("График дисперсии успешно сохранен!")
except Exception as e:
    print("Не удалось сохранить график в папку, сохраняю на диск C...")
    plt.savefig(r'C:\plot_variance.png', dpi=150)

plt.close()

print("\n5. Снижение размерности до оптимального числа...")
k = 10
# ... (остальной код до конца)

print("\n5. Снижение размерности до оптимального числа...")
k = 10
pca_optimal = PCA(n_components=k)
X_optimal = pca_optimal.fit_transform(X_processed)
print(f"Размерность до PCA: {X_processed.shape[1]} признаков")
print(f"Размерность после оптимального PCA: {X_optimal.shape[1]} компонент")
print(f"Суммарная доля сохранённой дисперсии: {pca_optimal.explained_variance_ratio_.sum() * 100:.2f}%")

print("\nГОТОВО! Все графики сохранены как картинки в папке с проектом.")