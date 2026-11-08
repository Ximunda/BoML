"""Naive bayes - алгоритм классификации который использует теорему Байеса для оценки вероятности того
что объект принадлежит к каждому из возможных классов (наивный потому что считает что признаки независимы друг от
друга при условии что уже известен класс, хотя на самом деле это не так и признаки могут иметь зависимость).

В sklearn есть несколько вариантов:
- GaussianNB - предполагает, что числовые признаки внутри каждого класса имеют нормальное распределение.
- MultinomialNB - предназначен преимущественно для частот и счётчиков (например количества слов).
- BernoulliNB - подходит для бинарных признаков.
Следовательно, выбираем GaussianNB
"""
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Стандартная подготовка данных
df = pd.read_csv("../codon_usage.csv")

df = df.dropna(subset=["Kingdom"])

y = df["Kingdom"]
X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

X = pd.get_dummies(X, columns=["DNAtype"])

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Создаём модель
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", GaussianNB(
        var_smoothing = 5, # Добавочная величина к дисперсии выборки для стабилизации вычисления
        priors=None         # Можно задать вероятности классов самому, но лучше оставить None, так программа
                            # посчитает сама, а сидеть и задавать всем классам вероятности не самое весёлое занятие
                            # так что делать этого я не буду, просто оставлю как показатель того что это возможно.
    ))
])

# Обучаем модель
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)

print("Точность Naive Bayes:", accuracy)
