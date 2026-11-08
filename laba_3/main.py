from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

def big_f(model):
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

def load_data():
    """Загрузка и подготовка датасета."""

    df = pd.read_csv("../codon_usage.csv")
    df = df.dropna(subset=["Kingdom"])

    y = df["Kingdom"]

    X = df.drop(columns=["Kingdom", "SpeciesName", "SpeciesID"])

    if "DNAtype" in X.columns:
        X = pd.get_dummies(X, columns=["DNAtype"], dtype=int)

    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y

DEFAULT_PARAMS = {
    "tree": {
        "max_depth": 15,
        "min_samples_split": 2,
        "criterion": "gini"
    },

    "forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "max_features": "sqrt"
    },

    "nb": {
        "var_smoothing": 5,
        "priors": None
    }
}

MODEL_NAMES = {
    "tree": "Decision Tree",
    "forest": "Random Forest",
    "nb": "Naive Bayes"
}


def build_model(model_name, params=None):
    """Создание модели с заданными параметрами."""

    if params is None:
        params = DEFAULT_PARAMS[model_name].copy()

    if model_name == "tree":
        classifier = DecisionTreeClassifier(
            **params,
            random_state=42
        )

    elif model_name == "forest":
        classifier = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=1
        )

    elif model_name == "nb":
        classifier = GaussianNB(**params)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier)
    ])

def big_F(model, X, y, cv):
    """Проверка модели с помощью 10-fold cross-validation."""

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print(f"Средняя точность: {scores.mean():.4f}")
    print(f"Стандартное отклонение: {scores.std():.4f}")

    return scores

def get_param_grid(model_name, y):

    if model_name == "tree":
        return {
            "classifier__max_depth": [5, 10, 15, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__criterion": ["gini", "entropy"]
        }

    elif model_name == "forest":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 15, None],
            "classifier__max_features": ["sqrt", "log2", None]
        }

    elif model_name == "nb":
        classes = np.unique(y)
        equal_priors = [1 / len(classes)] * len(classes)

        return {
            "classifier__var_smoothing": [
                1e-12, 1e-9, 1e-6, 1e-3, 1, 5
            ],
            "classifier__priors": [None, equal_priors]
        }

def tune_model(model_name, X, y, cv):
    """Поиск лучших гиперпараметров через GridSearchCV."""

    print(f"\nПодбор параметров: {MODEL_NAMES[model_name]}")

    model = build_model(model_name)

    search = GridSearchCV(
        estimator=model,
        param_grid=get_param_grid(model_name, y),
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X, y)

    print("\nЛучшие параметры:")
    for name, value in search.best_params_.items():
        print(f"{name.replace('classifier__', '')}: {value}")

    print(f"\nЛучшая средняя точность: {search.best_score_:.4f}")
    print(
        "Стандартное отклонение: "
        f"{search.cv_results_['std_test_score'][search.best_index_]:.4f}"
    )

    return search

def manual_params(model_name, y):

    params = DEFAULT_PARAMS[model_name].copy()

    print(f"\nНастройка: {MODEL_NAMES[model_name]}")
    print("Нажмите Enter, чтобы оставить значение по умолчанию.")

    if model_name == "tree":

        value = input(f"max_depth [{params['max_depth']}]: ").strip()
        if value:
            params["max_depth"] = (
                None if value.lower() == "none" else int(value)
            )

        value = input(
            f"min_samples_split [{params['min_samples_split']}]: "
        ).strip()
        if value:
            params["min_samples_split"] = int(value)

        value = input(
            f"criterion [{params['criterion']}] (gini/entropy): "
        ).strip()
        if value:
            params["criterion"] = value

    elif model_name == "forest":

        value = input(
            f"n_estimators [{params['n_estimators']}]: "
        ).strip()
        if value:
            params["n_estimators"] = int(value)

        value = input(
            f"max_depth [{params['max_depth']}]: "
        ).strip()
        if value:
            params["max_depth"] = (
                None if value.lower() == "none" else int(value)
            )

        value = input(
            f"max_features [{params['max_features']}] "
            "(sqrt/log2/none): "
        ).strip().lower()

        if value:
            if value in ["sqrt", "log2"]:
                params["max_features"] = value
            elif value == "none":
                params["max_features"] = None
            else:
                params["max_features"] = float(value)

    elif model_name == "nb":

        value = input(
            f"var_smoothing [{params['var_smoothing']}]: "
        ).strip()
        if value:
            params["var_smoothing"] = float(value)

        print("\nВыбор priors:")
        print("1. Автоматический расчёт (None)")
        print("2. Равные вероятности всех классов")
        print("3. Ввести вероятности самостоятельно")

        choice = input("Ваш выбор [1]: ").strip() or "1"

        classes = np.unique(y)

        if choice == "2":
            params["priors"] = [1 / len(classes)] * len(classes)

        elif choice == "3":
            print("\nВведите вероятность для каждого класса.")
            print("Сумма всех вероятностей должна равняться 1.")

            while True:
                priors = []

                for class_name in classes:
                    probability = float(
                        input(f"Класс {class_name}: ")
                    )
                    priors.append(probability)

                if (
                    all(0 <= p <= 1 for p in priors)
                    and np.isclose(sum(priors), 1.0)
                ):
                    params["priors"] = priors
                    break

                print("Ошибка: вероятности должны быть от 0 до 1,")
                print("а их сумма должна равняться 1. Попробуйте снова.")

        else:
            params["priors"] = None

    return params

def main():

    # Подготовка данных
    X, y = load_data()

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    print("===================================")
    print("       ВЫБОР МОДЕЛИ")
    print("===================================")

    print("1. Decision Tree")
    print("2. Random Forest")
    print("3. Naive Bayes")
    print("4. Все модели")

    model_choice = input("\nВыберите модель: ").strip()

    if model_choice == "1":
        selected_models = ["tree"]

    elif model_choice == "2":
        selected_models = ["forest"]

    elif model_choice == "3":
        selected_models = ["nb"]

    elif model_choice == "4":
        selected_models = ["tree", "forest", "nb"]

    else:
        print("Неверный выбор модели.")
        return

    print("\n===================================")
    print("       ВЫБОР РЕЖИМА")
    print("===================================")

    print("1. Автоматический подбор гиперпараметров")
    print("2. Ручной ввод гиперпараметров")
    print("3. Параметры по умолчанию")

    mode = input("\nВыберите режим: ").strip()

    if mode not in ["1", "2", "3"]:
        print("Неверный выбор режима.")
        return

    # Автоматический подбор гиперпараметров
    if mode == "1":

        all_results = []

        for model_name in selected_models:

            search = tune_model(
                model_name,
                X,
                y,
                cv
            )

            # Сохраняем полные результаты поиска,
            # только если выбраны все модели.
            if model_choice == "4":

                results = pd.DataFrame(search.cv_results_)

                results = results[[
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "rank_test_score"
                ]].copy()

                results.insert(
                    0,
                    "model",
                    MODEL_NAMES[model_name]
                )

                results["is_best"] = (
                    results["rank_test_score"] == 1
                )

                all_results.append(results)

        # Запись результатов всех моделей в один CSV
        if model_choice == "4":

            final_results = pd.concat(
                all_results,
                ignore_index=True
            )

            final_results = final_results.sort_values(
                ["model", "rank_test_score"]
            )

            filename = "hyperparameter_results.csv"

            final_results.to_csv(
                filename,
                index=False,
                encoding="utf-8-sig"
            )

            print(f"\nРезультаты сохранены в файл: {filename}")

    # Ручной ввод параметров
    elif mode == "2":

        for model_name in selected_models:

            params = manual_params(model_name, y)

            print("\nИспользуемые параметры:")
            print(params)

            model = build_model(model_name, params)

            print(f"\nПроверка {MODEL_NAMES[model_name]}:")

            big_F(model, X, y, cv)

    # Параметры по умолчанию
    elif mode == "3":

        for model_name in selected_models:

            params = DEFAULT_PARAMS[model_name].copy()

            print(f"\nПроверка {MODEL_NAMES[model_name]}")
            print("Параметры:", params)

            model = build_model(model_name, params)

            big_F(model, X, y, cv)


if __name__ == "__main__":
    main()
