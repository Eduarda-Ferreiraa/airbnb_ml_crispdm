# aibnb_ml_crispdm


# Business Understanding
This project analyzes **Airbnb listings in New York City**, one of the largest tourism markets in the world.  
NYC hosts millions of visitors each year (65M before 2019; 97% recovered in 2024), driving strong accommodation demand — but also challenges such as **short-term rental regulation** (85% of listings lack a license).  

## Business Problem
- Competition with hotels and platforms like Booking  
- Service consistency, regulatory risks, and pricing challenges  

## Objectives
1. **Price Optimization** — predict nightly rates to maximize host revenue and occupancy  
2. **Success Identification** — classify listings by performance (occupancy, reviews)  
3. **Market Segmentation** — cluster similar properties (budget, premium, family, etc.)  

## Success Criteria
- +10% average nightly revenue  
- +15% occupancy rate  
- ≥4 market segments covering 80% of listings  

## Risks
- Outdated data → cross-validate with external sources  
- Host resistance → demonstrate impact via case studies  

# Data Understanding

- Dataset: **48,895 listings**, **16 features**  
- Each row = one property (location, host, price, reviews, availability)  

## Key Attributes
- `id`, `host_id`, `host_name`  
- `neighbourhood_group`, `neighbourhood`  
- `latitude`, `longitude`  
- `room_type`  
- `price`  
- `minimum_nights`  
- `number_of_reviews`, `reviews_per_month`  
- `availability_365`  

## Issues
- Missing: `host_name`, `last_review`, `reviews_per_month`  
- Outliers: `price` up to $10k, `minimum_nights` up to 1250  
- Class imbalance: shared rooms underrepresented (~1k listings)  

## EDA Highlights
- **Price**: skewed; most $50–150, extreme outliers  
- **Room Type**: entire home/apt (25k), private room (22k), shared (1k)  
- **Availability**: bimodal (0 or 365 days)  
- **Location**: Manhattan + Brooklyn = ~84% of listings  

# Data Preparation

- Drop irrelevant columns (`name`, `host_name`, `last_review`)  
- Handle missing `reviews_per_month` (0 or median fill)  
- Outlier treatment: IQR clipping; min price = $10  
- Scale numeric features with **StandardScaler**  
- Encode categorical: one-hot for `room_type` and `neighbourhood_group`; top-20 neighborhoods kept  
- Drop IDs; select features via Spearman correlation (|corr| > 0.1)  

# Modelling

## 1. Price Prediction (Regression)  
- Models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, MLP  
- Best: **Random Forest** (nonlinear patterns), followed by MLP  
- Key factors: room type, location  

## 2. Success Classification  
- Target: high-value listings (entire apts in Manhattan, >$0, low availability)  
- Balancing: SMOTE + stratified split  
- Models: Logistic Regression, Random Forest, Gradient Boosting  
- Best: **Random Forest (GridSearchCV)** → F1 = 0.74  

## 3. Market Segmentation (Clustering)  
- Algorithm: K-Means (k=4 via elbow method)  
- Clusters:  
  - Budget (Brooklyn/Bronx, low prices, high availability)  
  - Luxury (Manhattan, high prices, low availability)  
  - Professional Hosts (multiple listings, diverse types)  
  - Budget (outside Manhattan, low prices, wide supply)  
- PCA used for 2D visualization  

All models built with **pipelines** (preprocessing + model) for reproducibility.  

# Deployment

A deployment script simulates predictions with final models:  

- **Price Prediction:** Random Forest  
- **Success Classification:** Random Forest + SMOTE  
- **Market Segmentation:** K-Means (k=4)  

## Workflow
1. Load models  
2. Sample a listing from processed dataset  
3. Prepare input (remove target-leak features)  
4. Output predictions: price, success (Yes/No), cluster  

Supports future integration with **Streamlit dashboards** or APIs.  

# Results & Conclusion

- **Regression:** accurate, consistent price forecasts  
- **Classification:** F1-score >70% (RF = 0.74 for success)  
- **Segmentation:** 100% listings grouped into 4 distinct profiles (target exceeded)  
- **Business Impact:** clearer pricing strategy, improved revenue/occupancy potential  
- **Usability:** interpretable metrics and visualizations for hosts & managers  
