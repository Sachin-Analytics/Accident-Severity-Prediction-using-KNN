# 🚦 Road Accident Severity Prediction using KNN

Multi-class classification project that predicts **accident severity** (`Slight Injury`, `Serious Injury`, `Fatal injury`) from driver, vehicle, road, weather and collision details using **K-Nearest Neighbors**.

## Problem
Severe and fatal accidents are rare but matter most. The dataset is heavily imbalanced, so plain accuracy is misleading. This project compares four KNN setups to show the accuracy vs. minority-class-detection trade-off.

## Dataset
- 12,316 rows × 21 columns (20 features + target `Accident_severity`), all categorical
- File: [`data/transportation_accident_knn_cleaned.csv`](data/transportation_accident_knn_cleaned.csv)
- Target distribution: Slight ~84.6% | Serious ~14.2% | Fatal ~1.3%

## Workflow
1. Load data and explore
2. Mode imputation for missing values (11 columns)
3. Remove 1 duplicate row (12,316 → 12,315)
4. Feature engineering: extract `Hour` from `Time`
5. EDA (severity, day, age, vehicle, road, light, weather, collision type, hour)
6. Label encoding + stratified 80/20 split + `StandardScaler`
7. KNN tuning for K = 1…10 across 4 configurations:
   baseline, distance-weighted, SMOTE, SMOTENC
8. Model comparison and selection

## Results

| Model | Accuracy | Macro F1 | Fatal Recall | Serious Recall |
|---|---|---|---|---|
| KNN (K=10) | 84.00% | 0.31 | 0.00 | 0.00 |
| KNN + Distance weighting | 84.29% | 0.31 | 0.00 | 0.00 |
| KNN + SMOTE | 40.72% | 0.26 | 0.12 | 0.46 |
| KNN + SMOTENC | 54.77% | 0.31 | 0.16 | 0.34 |

**Takeaway:** the highest-accuracy models never detect Fatal/Serious cases (they just predict the majority class). **KNN + SMOTENC** gives the best balance between accuracy and catching severe accidents.

## Project structure
```
├── data/        # cleaned dataset (CSV)
├── notebooks/   # accident_knn.ipynb
├── reports/     # project presentation (PPTX)
├── requirements.txt
└── README.md
```

## Run it
```bash
git clone https://github.com/<your-username>/road-accident-severity-knn.git
cd road-accident-severity-knn
pip install -r requirements.txt
jupyter notebook notebooks/accident_knn.ipynb
```

## Tech stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · imbalanced-learn

## Possible improvements
- Try tree-based models (Random Forest, XGBoost) which handle categorical data and imbalance better
- One-hot / target encoding instead of label encoding (KNN assumes distance between codes is meaningful)
- Optimize for recall / F1 of severe classes, tune K beyond 10

## Author
**Sachin More** · [Medium](https://medium.com/@moresachin9124)
