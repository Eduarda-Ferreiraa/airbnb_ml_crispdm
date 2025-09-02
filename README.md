# aibnb_ml_crispdm

## Business Understanding
O projeto analisa dados do **Airbnb em Nova Iorque**, um dos maiores mercados turísticos do mundo.   
Nova Iorque recebe milhões de visitantes por ano (65M antes de 2019; 97% desse recorde recuperado em 2024), o que gera forte procura por alojamento — mas também desafios como **regulação de alugueres de curto prazo** (85% das propriedades listadas não têm licença).  

**Problema de Negócio**  
- Competição direta com hotéis e plataformas como Booking. 
- Consistência dos serviços, risco regulatório, precificação adequada.  

**Objetivos do Projeto**  
1. **Otimizar a Precificação** — prever o preço por noite para ajudar anfitriões a maximizar receita e ocupação.  
2. **Identificar Listagens de Sucesso** — categorizar propriedades com base no desempenho (ocupação, reviews).  
3. **Segmentar o Mercado** — agrupar propriedades semelhantes (económicas, premium, familiares, etc.).  

**Critérios de Sucesso**  
- Aumentar receita média por noite em **10%** com previsões de preço.  
- Elevar taxa de ocupação em **15%**.  
- Identificar pelo menos **4 segmentos de mercado** representando 80% das listagens.   

**Riscos e Contingências**  
- Dados desatualizados → cruzar com fontes externas.  
- Resistência dos anfitriões → mostrar impacto positivo via estudos de caso.  

## Data Understanding
- Dataset do **Airbnb em Nova Iorque** com **48.895 registos** e **16 atributos**.  
- Cada linha representa uma propriedade (listagem) com informações de **localização, preço, disponibilidade, anfitrião e avaliações**.  

**Principais Atributos**  
- `id`: identificador único da listagem  
- `host_id`, `host_name`: identificador e nome do anfitrião  
- `neighbourhood_group`, `neighbourhood`: distrito e bairro  
- `latitude`, `longitude`: localização geográfica  
- `room_type`: tipo de alojamento (entire home/apt, private room, shared room)  
- `price`: preço por noite (USD)  
- `minimum_nights`: noites mínimas exigidas  
- `number_of_reviews`, `reviews_per_month`: popularidade da listagem  
- `availability_365`: disponibilidade em dias por ano  

**Valores Ausentes**  
- `host_name`, `last_review`, `reviews_per_month` apresentam missing values → requerem tratamento.  

**Análise Exploratória (EDA)**  
- **Preço**: distribuição altamente assimétrica; maioria entre **$50–150/noite**; outliers até **$10.000**.  
- **Room Type**:  
  - Entire home/apt → 25k listagens (mais comum)  
  - Private room → 22k  
  - Shared room → ~1k (pouco representado, dados desbalanceados)  
- **Disponibilidade**: bimodal, com picos em **0 dias** (15k listagens) e **365 dias** (5k listagens).  
- **Localização**: Manhattan e Brooklyn concentram **~84%** das listagens; Bronx e Staten Island quase irrelevantes.  

**Análise Multivariada**  
- **Preço x Reviews**: correlação muito fraca (−0.036). Preços altos não geram mais reviews.  
- **Preço x Disponibilidade**: fraca correlação positiva (+0.078). Listagens caras ficam mais disponíveis (menos reservas).  
- **Reviews Totais x Reviews/Mês**: correlação moderada (+0.55).  
- **Disponibilidade x Popularidade**: fraca correlação positiva (~0.19).  

**Problemas Identificados**  
- **Outliers severos** em `price` (até $10.000) e `minimum_nights` (até 1250 noites).  
- **Desbalançamento** de `room_type` (shared rooms quase inexistentes).  
- **Valores ausentes** relevantes (`reviews_per_month` com +10k missing).  

### Data Preparation

**Passos de Pré-processamento**  
- **Análise inicial:** verificação do tamanho do dataset, colunas, estatísticas descritivas e valores em falta.  
- **Tratamento de valores omissos:**  
  - Remoção de colunas irrelevantes (`name`, `host_name`, `last_review`).  
  - `reviews_per_month`:  
    - Definido como **0** para listagens sem reviews.  
    - Valores faltantes preenchidos com a **mediana** (robusta a outliers).  
- **Outliers:**  
  - Detectados e corrigidos via **IQR clipping** para `price` e `minimum_nights`.  
  - Imposto mínimo realista de **$10** no preço.  
- **Normalização:**  
  - Variáveis numéricas (`price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `availability_365`) padronizadas com **StandardScaler**.  
- **Codificação categórica:**  
  - `neighbourhood_group` e `room_type` → **One-Hot Encoding**.  
  - `neighbourhood`: top 20 bairros mantidos, restantes agrupados em **“Outros”**.  
- **Seleção de atributos:**  
  - Removidos IDs (`id`, `host_id`).  
  - Seleção baseada na **correlação de Spearman** com `price` (|corr| > 0.1).  

**Justificação**  
- Mantida a integridade dos dados sem eliminar linhas.  
- Redução da dimensionalidade → evita overfitting e melhora performance dos modelos.  
- Preparação garante dados balanceados, escalados e compatíveis com ML.  

## Modelling

Foram desenvolvidos modelos para **três objetivos distintos**: previsão de preços, identificação de listagens de sucesso e segmentação de mercado.  

### Previsão de Preços (Regressão)
- **Modelos testados:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, MLP.  
- **Validação:** 5-fold cross-validation, métricas RMSE, MAE e R².  
- **Resultados:**  
  - **Random Forest** → melhor desempenho (captura relações não lineares).  
  - **MLP** → resultados competitivos.  
  - **Linear Regression** → limitado (relação não linear entre variáveis).  
- **Interpretação:**  
  - Room type e localização foram os principais fatores de decisão.  
  - Ajuste de hiperparâmetros via GridSearchCV.  
  - Importância das variáveis analisada no modelo final.  

### Identificação de Listagens de Sucesso (Classificação)
- **Definição da variável alvo:** listagens de maior valor (aptos inteiros em Manhattan, preço > 0, baixa disponibilidade).  
- **Cuidados:** exclusão de features ligadas à definição de sucesso (para evitar data leakage).  
- **Pré-processamento:**  
  - **SMOTE** para balancear classes (success ≪ insuccess).  
  - Divisão treino/teste com estratificação.  
- **Modelos testados:** Logistic Regression, Random Forest, Gradient Boosting.  
- **Resultados:**  
  - **Random Forest (com GridSearchCV)** → melhor equilíbrio entre F1-score e recall.  
  - Logistic Regression → recall muito alto, mas muitos falsos positivos.  
  - Gradient Boosting → bom desempenho, mas abaixo do RF otimizado.  
- **Interpretação:** variáveis mais relevantes foram **room type, localização e disponibilidade**.  

### Segmentação de Mercado (Clustering)
- **Algoritmo:** K-Means.  
- **Seleção de features:** localização (latitude/longitude), preço, disponibilidade, tipo de quarto e nº de listagens do host.  
- **Determinação de k:** método do cotovelo → k = 4.  
- **Clusters identificados:**  
  - **Cluster 0 – Económico (Brooklyn/Bronx, preços baixos, alta disponibilidade).**  
  - **Cluster 1 – Luxo (Manhattan, preços elevados, baixa disponibilidade).**  
  - **Cluster 2 – Hosts Profissionais (múltiplas listagens, diversidade de tipologias).**  
  - **Cluster 3 – Económico (alta oferta, preços mais baixos, fora de Manhattan).**  
- **Visualização:** PCA para projeção bidimensional → clusters bem separados.  

Em todas as tarefas foram usados pipelines integrando **pré-processamento + modelo**, garantindo reprodutibilidade e comparações justas.

## Deployment

Após a modelação e avaliação, foi desenvolvido um **script de deployment** para simular previsões com os modelos finais exportados.  

### Objetivos simulados:
- **Previsão de preço**: utilizando o Random Forest otimizado com GridSearch (pipeline completo).  
- **Classificação de sucesso**: com Random Forest balanceado via SMOTE.  
- **Segmentação de mercado**: atribuição de clusters através do KMeans (k=4).  

### Funcionamento do script:
1. Carregamento dos modelos finais.  
2. Seleção aleatória de uma linha do dataset tratado (nova listagem).  
3. Preparação da entrada para cada modelo (remoção de atributos usados na variável alvo).  
4. Geração das previsões:  
   - Preço previsto (normalizado).  
   - Sucesso previsto (Sim/Não).  
   - Cluster atribuído (segmento de mercado).  

O processo garante **robustez**, lidando com diferenças entre colunas do treino e da predição.  
 Esta fase valida o funcionamento prático dos modelos e abre espaço para extensões, como **interfaces gráficas (Streamlit)** ou APIs web.  

## Conclusão

Os objetivos definidos na fase inicial foram **atingidos com sucesso**:  

- **Precisão dos modelos**:  
  - Classificação → F1-score acima de 70% (Random Forest alcançou 0.74 para a classe de sucesso).  
  - Regressão → baixo erro e previsões consistentes.  
- **Impacto na precificação**: previsões claras e robustas, com potencial de apoiar anfitriões a definir preços competitivos.  
- **Segmentação eficaz**: 100% das listagens agrupadas em **4 perfis distintos**, superando a meta de 80%.  
- **Compreensibilidade**: métricas e visualizações intuitivas permitem uso prático por gestores e anfitriões.  

