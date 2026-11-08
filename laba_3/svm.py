"""Алгоритм, который проводит границу (строит гиперплоскость) между
объектами классов на максимальном расстоянии от них используя метод опорных векторов.
Опорными векторами называются ближайшие к гиперплоскости элементы классов, ориентируясь на них корректируется
гиперплоскость так, чтобы увеличить расстояние от этих точек.
"""
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Загружаем датасет
df = pd.read_csv("codon_usage.csv")

# Удаляем строки без целевого класса
df = df.dropna(subset=["Kingdom"])

# Разделяем признаки и целевую переменную
y = df["Kingdom"]
X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

# Кодируем категориальный признак DNAtype
X = pd.get_dummies(X, columns=["DNAtype"])

# Преобразуем признаки в числовой формат
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Создаём модель SVM
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        C = 1.0,            # Штраф за ошибку классификации, меньше штраф - проще гиперплоскость,
                            # больше штраф - выше шанс переобучения.
        kernel="rbf",       # Форма гиперплоскости. Linear - линейная граница, rbf - нелинейная граница,
                            # poly - полиномиальная, sigmoid - граница на основе сигмоидальной функции.
        gamma="scale"       # Влияние отдельных объектов на форму границы, условно то насколько сложную
                            # фигуру построит алгоритм, низкое значение - "мягкая" форма повторяющая
                            # форму границы классов плавно, высокое значение - "жесткая" форма,
                            # как-бы "плотно" огибающая оба класса. "scale" значит автоматический расчёт значения.
    ))
])

# Обучаем модель
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)

print("Точность SVM:", accuracy)
