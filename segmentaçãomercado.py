import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Carregar dados
df = pd.read_csv("dataset_tratado.csv")

# 2. Selecionar features relevantes para clustering
features = [
    'availability_365', 'calculated_host_listings_count', 'latitude', 'longitude', 'price',
    'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room',
    'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn',
    'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens'
]

X = df[features].copy()

# 3. Normalizar as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Método do cotovelo para escolher o número de clusters
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia, 'o-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

# 5. Ajustar o número de clusters (exemplo: 4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Perfil dos clusters
print("\nNúmero de listagens por cluster:")
print(df['cluster'].value_counts())

print("\nMédias por cluster:")
print(df.groupby('cluster')[features].mean())

# 7. Visualização dos clusters com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
for i in range(k):
    plt.scatter(X_pca[df['cluster']==i, 0], X_pca[df['cluster']==i, 1], label=f'Cluster {i}', alpha=0.5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters de Mercado (PCA)')
plt.legend()
plt.tight_layout()
plt.show()

import joblib

# Exportar o modelo KMeans final
joblib.dump(kmeans, "modelo_cluster.pkl")
