""""""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# pd.set_option('display.max_columns', None)

df = pd.read_csv("../codon_usage.csv")

print(df.head())
print(df.shape)
print(df.columns)


# Разделяем выборку на Kingdom который мы определяем
# и остальное (кроме SpeciesName и SpeciesID так как они не участвуют в определении)

df = df.dropna(subset=["Kingdom"])
y = df["Kingdom"]
X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

# Подготовка данных так как DT не умеет обучиться на текстовых категориях то
# преобразуем в числовые столбцы (One-Hot encoding)
X = pd.get_dummies(X, columns=["DNAtype"])

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.fillna(X.median())

print(X)

# Разделим данные на тестовую и обучающую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=2,
        criterion="gini",  # Критерий оценки качества разделения узлов
        random_state=42
    ))
])

# sklearn сам строит дерево решений и сам подбирает признаки для разделения
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Точность:", accuracy)

# Визуализация
plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.tight_layout()
# plt.show()