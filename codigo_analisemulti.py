import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('/Users/eduardaferreira/Desktop/AB_NYC_2019.csv')
sns.set(style="whitegrid")

##5
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='price', y='number_of_reviews', hue='room_type', size='availability_365', 
                sizes=(20, 200), alpha=0.5, palette='deep')
plt.xlim(0, 2000)  # Limitando para melhor visualização, conforme descrito
plt.title('Preço vs. Número de Revisões por Tipo de Quarto e Disponibilidade')
plt.xlabel('Preço por Noite (USD)')
plt.ylabel('Número de Revisões')
plt.legend(title='Tipo de Quarto')
plt.show()

##6
plt.figure(figsize=(10, 8))
# Selecionar as colunas numéricas para a matriz de correlação
numeric_cols = ['price', 'number_of_reviews', 'reviews_per_month', 'availability_365', 
                'minimum_nights', 'calculated_host_listings_count']
correlation_matrix = data[numeric_cols].corr()

# Criar o heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.show()