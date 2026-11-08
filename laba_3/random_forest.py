"""Множество деревьев решений которые принимают решения независимо, а затем демократия

Основные гиперпараметры:
n_estimators        - Количество деревьев
max_depth           - Максимальная глубина деревьев
max_features        - Количество признаков для поиска разделения
min_samples_split   - Минимальное количество объектов для разделения
min_samples_leaf    - Минимальное количество объектов в листе
criterion           - Критерий разделения
bootstrap           - Использовать ли случайную выборку с возвращением
max_samples         - Количество объектов для обучения каждого дерева
class_weight        - Веса классов
max_leaf_nodes      - Максимальное количество листьев
ccp_alpha           - Параметр обрезки деревьев
oob_score           - Оценка качества на объектах, не попавших в bootstrap-выборку
n_jobs              - Количество используемых процессорных ядер
"""
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Загружаем данные
df = pd.read_csv("../codon_usage.csv")

# Удаляем строки без целевого класса
df = df.dropna(subset=["Kingdom"])

# Выделяем целевой признак и остальные признаки
y = df["Kingdom"]
X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

# Преобразуем DNAtype в числовые признаки
X = pd.get_dummies(X, columns=["DNAtype"])

# Преобразуем остальные значения в числа
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")


# Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # Размер тестовой выборки 20%. Обучающая соответственно - 80%
    random_state=42,
    stratify=y
)


# Создаём Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100, # количество деревьев в лесу
        max_depth=15,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1 # использовать все ядра процессора
    ))
])


# Обучаем модель
model.fit(X_train, y_train)

# Получаем предсказания
y_pred = model.predict(X_test)

# Вычисляем точность
accuracy = accuracy_score(y_test, y_pred)

print("Точность Random Forest:", accuracy)