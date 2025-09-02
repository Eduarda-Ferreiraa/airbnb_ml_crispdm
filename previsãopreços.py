import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Carregar dados
df = pd.read_csv("dataset_tratado.csv")
if 'id' in df.columns: df = df.drop(columns=['id'])

X = df.drop(columns=["price"])
y = df["price"]

# 2. Identificar colunas numéricas e categóricas
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

# 3. Pré-processamento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 4. Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Modelos principais (incluindo Decision Tree com max_depth=4 para referência)
modelos = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "MLP (Rede Neural)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "Decision Tree (max_depth=4)": DecisionTreeRegressor(max_depth=4, random_state=42)  # Para referência/explicação
}

# 6. Avaliação dos modelos principais
print("\n=== Avaliação dos Modelos Principais ===")
for nome, modelo in modelos.items():
    pipe = Pipeline([
        ('pre', preprocessor),
        ('reg', modelo)
    ])
    # Validação cruzada
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    print(f"{nome} - RMSE médio (cross-val): {abs(scores.mean()):.2f} +/- {scores.std():.2f}")
    # Treinar e avaliar no teste
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{nome} - Teste RMSE: {rmse:.2f}")
    print(f"{nome} - Teste MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"{nome} - Teste R²: {r2_score(y_test, y_pred):.2f}\n")

# 7. Ajuste de hiperparâmetros para Random Forest
param_grid_rf = {
    'reg__n_estimators': [100, 200],
    'reg__max_depth': [None, 10, 20]
}
rf_pipe = Pipeline([
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(random_state=42))
])
grid_rf = GridSearchCV(rf_pipe, param_grid_rf, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Melhores parâmetros Random Forest:", grid_rf.best_params_)

# 8. Visualização para o melhor modelo (Random Forest)
y_pred_rf = grid_rf.predict(X_test)
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred_rf, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal (y=x)')
plt.xlabel("Preço Real")
plt.ylabel("Preço Previsto")
plt.title("Scatter Plot - Preço Real vs Previsto (Random Forest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

errors_rf = y_test - y_pred_rf
plt.figure(figsize=(8,5))
plt.hist(errors_rf, bins=40, edgecolor='black')
plt.xlabel("Erro (Real - Previsto)")
plt.ylabel("Frequência")
plt.title("Histograma dos Erros - Random Forest")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Importância das variáveis do melhor Random Forest
best_rf = grid_rf.best_estimator_.named_steps['reg']
if hasattr(best_rf, "feature_importances_"):
    feature_names = []
    if num_cols:
        feature_names += num_cols
    if cat_cols:
        encoder = grid_rf.best_estimator_.named_steps['pre'].named_transformers_['cat']
        feature_names += encoder.get_feature_names_out(cat_cols).tolist()
    importances = best_rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 variáveis mais importantes (Random Forest):")
    print(feat_imp.head(10))
    feat_imp.head(10).plot(kind='barh')
    plt.title("Top 10 Features - Random Forest (GridSearch)")
    plt.show()

# 10. TESTAR VÁRIOS max_depth NA DECISION TREE (GridSearchCV)
print("\n=== GridSearchCV para Decision Tree ===")
dt_pipe = Pipeline([
    ('pre', preprocessor),
    ('reg', DecisionTreeRegressor(random_state=42))
])
param_grid_dt = {
    'reg__max_depth': [2, 3, 4, 5, 6, 8, 10, 12, None]
}
grid_dt = GridSearchCV(
    dt_pipe,
    param_grid_dt,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
grid_dt.fit(X_train, y_train)

print("Melhor max_depth:", grid_dt.best_params_['reg__max_depth'])
print("Melhor RMSE (cross-val):", abs(grid_dt.best_score_))

# Avaliação no conjunto de teste com a melhor Decision Tree
y_pred_dt = grid_dt.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print(f"Decision Tree (melhor max_depth) - Teste RMSE: {rmse_dt:.2f}")

# 11. VISUALIZAÇÃO DA ÁRVORE RASA PARA EXPLICAÇÃO (max_depth=4)
dt_pipe_explic = Pipeline([
    ('pre', preprocessor),
    ('reg', DecisionTreeRegressor(max_depth=4, random_state=42))
])
dt_pipe_explic.fit(X_train, y_train)
dt_model_explic = dt_pipe_explic.named_steps['reg']

feature_names = []
if num_cols:
    feature_names += num_cols
if cat_cols:
    encoder = dt_pipe_explic.named_steps['pre'].named_transformers_['cat']
    feature_names += encoder.get_feature_names_out(cat_cols).tolist()

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model_explic,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão para Explicação (max_depth=4)")
plt.show()

import joblib

# Substitui pelo pipeline final (por exemplo, grid_rf se usaste GridSearch)
joblib.dump(grid_rf, "modelo_preco.pkl")
