""""""
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def big_F(model):
    """Функция получает модель и обучает её. Нужна только для удобства."""
    # Подготовка данных
    # pd.set_option('display.max_columns', None)
    df = pd.read_csv("../codon_usage.csv")

    # Разделяем выборку на Kingdom который мы определяем
    # и остальное (кроме SpeciesName и SpeciesID так как они не участвуют в определении)

    df = df.dropna(subset=["Kingdom"])
    y = df["Kingdom"]
    X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

    # Преобразуем в числовые столбцы (One-Hot encoding)
    X = pd.get_dummies(X, columns=["DNAtype"])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.median())

    # Разделим данные на тестовую и обучающую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Точность:", accuracy)


def tenV(model):
    """Функция получает модель и проверяет её с помощью 10-fold CV."""

    # Подготовка данных
    df = pd.read_csv("../codon_usage.csv")

    df = df.dropna(subset=["Kingdom"])
    y = df["Kingdom"]
    X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

    # One-Hot Encoding
    X = pd.get_dummies(X, columns=["DNAtype"])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Настраиваем 10-fold cross-validation
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    # Проводим проверку
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    # Выводим результаты
    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print("Средняя точность:", scores.mean())
    print("Стандартное отклонение:", scores.std())

def main():
    pass
if __name__ == "__main__":
    main()
