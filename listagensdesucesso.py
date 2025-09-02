import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 1. Carregar e preparar os dados
df = pd.read_csv("dataset_tratado.csv")

# 2. Criar a coluna de sucesso
df["success"] = (
    (df["room_type_Entire home/apt"] == 1) &
    (df["neighbourhood_group_Manhattan"] == 1) &
    (df["price"] > 0) &
    (df["availability_365"] < df["availability_365"].median())
).astype(int)

# 3. Remover as colunas que definem 'success' das features de entrada
X = df.drop(columns=[
    "success",
    "room_type_Entire home/apt",
    "neighbourhood_group_Manhattan",
    "price",
    "availability_365"
])
y = df["success"]

# 4. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Modelos com pipeline (SMOTE + StandardScaler para modelos que precisam)
modelos = {
    "Logistic Regression": Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42, n_estimators=100))
    ]),
    "Gradient Boosting": Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42, n_estimators=100))
    ])
}

# 6. Validação cruzada (5 folds)
print("\n=== Validação Cruzada (Cross-Validation) ===")
resultados_cv = {}
for nome, pipe in modelos.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    print(f"{nome} - F1-score médio (cross-val): {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    resultados_cv[nome] = scores

# 7. Ajuste de hiperparâmetros para Random Forest
print("\n=== GridSearchCV para Random Forest ===")
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5]
}
rf_pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Melhores parâmetros Random Forest:", grid.best_params_)

# 8. Treinamento final e avaliação no teste
print("\n=== Avaliação no Teste ===")
resultados = {}
for nome, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    print(f"\n--- {nome} ---")
    print(classification_report(y_test, y_pred))
    resultados[nome] = {
        "f1_score": f1,
        "recall": recall,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

# 9. Avaliação do melhor Random Forest encontrado no GridSearch
print("\n--- Random Forest (Melhor do GridSearch) ---")
y_pred_grid = grid.predict(X_test)
print(classification_report(y_test, y_pred_grid))

# 10. Matriz de Confusão para TODOS os modelos
print("\n=== Matrizes de Confusão para todos os modelos ===")
for nome, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fracasso", "Sucesso"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {nome}")
    plt.show()

# 11. Matriz de Confusão para o MELHOR Random Forest (GridSearch)
print("\n=== Matriz de Confusão - Melhor Random Forest (GridSearch) ===")
cm_grid = confusion_matrix(y_test, y_pred_grid)
disp_grid = ConfusionMatrixDisplay(confusion_matrix=cm_grid, display_labels=["Fracasso", "Sucesso"])
disp_grid.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Melhor Random Forest (GridSearch)")
plt.show()

# 12. Visualização dos resultados finais
labels = list(resultados.keys())
f1_scores = [resultados[m]["f1_score"] for m in labels]
recalls = [resultados[m]["recall"] for m in labels]

plt.figure(figsize=(10, 5))
bar_width = 0.35
x = np.arange(len(labels))
plt.bar(x, f1_scores, width=bar_width, label="F1-score")
plt.bar(x + bar_width, recalls, width=bar_width, label="Recall")
plt.xlabel("Modelos")
plt.ylabel("Pontuação")
plt.title("Comparação de F1-score e Recall por Modelo")
plt.xticks(x + bar_width / 2, labels)
plt.legend()
plt.tight_layout()
plt.show()

# 13. Importância das variáveis do melhor Random Forest
best_rf = grid.best_estimator_.named_steps['clf']
if hasattr(best_rf, "feature_importances_"):
    importances = best_rf.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 variáveis mais importantes (Random Forest):")
    print(feat_imp.head(10))
    feat_imp.head(10).plot(kind='barh')
    plt.title("Top 10 Features - Random Forest (GridSearch)")
    plt.show()

import joblib

# Substitui pelo melhor modelo encontrado (por exemplo, grid)
joblib.dump(grid, "modelo_sucesso.pkl")
