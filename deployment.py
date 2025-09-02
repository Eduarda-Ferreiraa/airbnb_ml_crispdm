import joblib
import pandas as pd

# 1. Carregar os modelos
modelo_preco = joblib.load("modelo_preco.pkl")
modelo_sucesso = joblib.load("modelo_sucesso.pkl")
modelo_cluster = joblib.load("modelo_cluster.pkl")

# 2. Carregar dataset tratado completo
df = pd.read_csv("dataset_tratado.csv")

# 3. Selecionar linha real
linha_real = df.sample(1, random_state=42)

# 4. Preparar inputs para cada modelo

## 4a. Para modelo de sucesso
entrada_sucesso = linha_real.drop(columns=[
    "success", "room_type_Entire home/apt",
    "neighbourhood_group_Manhattan", "price", "availability_365"
], errors='ignore')

## 4b. Para modelo de preço (usa tudo exceto targets)
entrada_preco = linha_real.drop(columns=["price", "success"], errors='ignore')

## 4c. Para modelo de clustering (apenas 12 colunas usadas no treino do KMeans)
features_cluster = [
    'availability_365', 'calculated_host_listings_count', 'latitude', 'longitude', 'price',
    'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room',
    'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
    'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens'
]
entrada_cluster = linha_real[features_cluster]

# 5. Previsões
preco_previsto = modelo_preco.predict(entrada_preco)[0]
sucesso_previsto = modelo_sucesso.predict(entrada_sucesso)[0]
cluster_previsto = modelo_cluster.predict(entrada_cluster)[0]

# 6. Resultados
print("\n Resultados da Simulação:")
print(f" Preço previsto: ${preco_previsto:.2f}")
print(f" Sucesso previsto: {'Sim' if sucesso_previsto == 1 else 'Não'}")
print(f" Segmento atribuído (Cluster): {cluster_previsto}")
