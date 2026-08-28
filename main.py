import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 1. Load & Clean
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Calculate ratio for scale_pos_weight: roughly (majority_class / minority_class)
# 73 / 27 ≈ 2.7
scale_weight = (len(y) - sum(y)) / sum(y)

# 2. Pipeline Setup
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Swap RandomForest for XGBoost
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42, 
        scale_pos_weight=scale_weight, 
        eval_metric='logloss',
        max_depth=4,       # Lower depth helps prevent overfitting
        learning_rate=0.05
    ))
])

# 3. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

# 4. Predict Probabilities instead of hard labels
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Tune Threshold: Instead of default 0.5, lower it to 0.4 to catch more churners
custom_threshold = 0.4
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

# 5. Evaluate
print(f"--- Updated Report (Threshold: {custom_threshold}) ---")
print(classification_report(y_test, y_pred_custom))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")


# 1. Save the entire pipeline for deployment later
joblib.dump(pipeline, 'churn_pipeline_model.pkl')
print("\n[INFO] Model successfully saved to 'churn_pipeline_model.pkl'")

# 2. Extract feature names from the preprocessor to see what matters
# Get encoded categorical feature names from the OneHotEncoder step
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))

# Combine with numeric features
all_features = numeric_features + encoded_cat_features

# 3. Get Feature Importances from XGBoost
importances = pipeline.named_steps['classifier'].feature_importances_

# Map them together into a clean DataFrame
feature_imp_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features Driving Churn ---")
print(feature_imp_df.head(5).to_string(index=False))