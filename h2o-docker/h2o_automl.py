"""
Assignment 3 - Question 9: H2O AutoML
Used professor's H2O_AutoML.ipynb notebook from Discussion in Canvas
"""
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np

# ============================================================
# 1. Initialize H2O
# ============================================================
h2o.init(max_mem_size="2G", nthreads=1)

# ============================================================
# 2. Load Data 
# ============================================================
df = pd.read_csv('athletes.csv')
print(f"Original shape: {df.shape}")
print(f"Medal distribution:\n{df['medal'].value_counts(dropna=False)}")

# Check what values are in medal column
print(f"\nUnique medal values: {df['medal'].unique()}")

# Handle both string "None" and NaN
df['won_medal'] = df['medal'].apply(lambda x: "No" if (pd.isna(x) or str(x).strip().lower() == "none") else "Yes")
df = df.drop(['name', 'medal', 'athlete_id'], axis=1)

print(f"\nTarget distribution:")
print(df['won_medal'].value_counts())

# Save to CSV and let H2O read it directly (avoids conversion issues)
df.to_csv('athletes_clean.csv', index=False)
hf = h2o.import_file('athletes_clean.csv')

# Verify columns and types
print(f"\nH2O Frame columns: {hf.columns}")
print(f"H2O Frame shape: {hf.shape}")
print(f"won_medal type: {hf['won_medal'].type}")

# Set target as factor
hf['won_medal'] = hf['won_medal'].asfactor()
print(f"won_medal levels: {hf['won_medal'].levels()}")
print(f"won_medal level count: {hf['won_medal'].nlevels()}")

# Set categorical columns
hf['country'] = hf['country'].asfactor()
hf['sport'] = hf['sport'].asfactor()

# Define features and target BEFORE splitting
y = "won_medal"
x = [col for col in hf.columns if col != y]

print(f"\nFeatures: {x}")
print(f"Target: {y}")

# Split data with stratification by using fold_column approach
train, test = hf.split_frame(ratios=[0.8], seed=42)

print(f"\nTraining set: {train.shape}")
print(f"Test set: {test.shape}")
print(f"Train target distribution:")
print(train['won_medal'].table())
print(f"Test target distribution:")
print(test['won_medal'].table())

# ============================================================
# 3. Run H2O AutoML - ALL FEATURES 
# ============================================================
print("RUNNING H2O AutoML - ALL FEATURES")
print("=" * 60)

aml_all = H2OAutoML(
    max_models=10,
    seed=42,
    exclude_algos=["StackedEnsemble", "DeepLearning"],
    verbosity="info",
    nfolds=5
)

aml_all.train(x=x, y=y, training_frame=train)

# ============================================================
# 4. Leaderboard - ALL FEATURES 
# ============================================================
print("Q9-Q5a: LEADERBOARD - ALL FEATURES")
print("=" * 60)

lb_all = aml_all.leaderboard
print("\nFull Leaderboard:")
print(lb_all.head(rows=10))

# Top 3 by validation score
print("\n" + "=" * 70)
print("TOP 3 MODELS BY VALIDATION SCORE (ALL FEATURES)")
print("=" * 70)
lb_df = lb_all.as_data_frame()
for i in range(min(3, len(lb_df))):
    print(f"\n#{i+1}: {lb_df.iloc[i]['model_id']}")
    for col in lb_df.columns[1:]:
        print(f"    {col}: {lb_df.iloc[i][col]}")

# ============================================================
# 5. Best Model Details 
# ============================================================
print("BEST MODEL DETAILS")
print("=" * 70)

# Predict on test set 
pred_all = aml_all.leader.predict(test)
print("\nPredictions (first 5):")
print(pred_all.head())

# Model performance 
print("\nModel Performance on Test Set:")
perf_all = aml_all.leader.model_performance(test)
print(perf_all)

# ============================================================
# 6. Feature Importance 
# ============================================================
print("DATA INSIGHTS & TOP 5 FEATURES")
print("=" * 70)

# Get model IDs 
model_ids = list(aml_all.leaderboard['model_id'].as_data_frame().iloc[:, 0])
print(f"\nAll model IDs: {model_ids}")

# Get feature importance
top_features = x  # default fallback
top_3_features = x[:3]

for mid in model_ids:
    try:
        m = h2o.get_model(mid)
        varimp = m.varimp(use_pandas=True)
        if varimp is not None and len(varimp) > 0:
            print(f"\nFeature importance from model: {mid}")
            print("\nTop 5 Features:")
            print(varimp.head(5))
            top_features = varimp['variable'].tolist()[:5]
            top_3_features = varimp['variable'].tolist()[:3]
            print(f"\nTop 5: {top_features}")
            print(f"Top 3: {top_3_features}")
            break
    except:
        continue

# ============================================================
# 7. Run H2O AutoML - TOP 3 FEATURES ONLY
# ============================================================
print("RUNNING H2O AutoML - TOP 3 FEATURES ONLY")
print("=" * 60)
print(f"Using features: {top_3_features}")

aml_top = H2OAutoML(
    max_models=10,
    seed=42,
    exclude_algos=["StackedEnsemble", "DeepLearning"],
    verbosity="info",
    nfolds=5
)

aml_top.train(x=top_3_features, y=y, training_frame=train)

# ============================================================
# 8. Leaderboard - TOP FEATURES
# ============================================================
print("Q9-Q5b: LEADERBOARD - TOP 3 FEATURES")
print("=" * 60)

lb_top = aml_top.leaderboard
print("\nFull Leaderboard:")
print(lb_top.head(rows=10))

print("TOP 3 MODELS BY VALIDATION SCORE (TOP 3 FEATURES)")
print("=" * 60)
lb_top_df = lb_top.as_data_frame()
for i in range(min(3, len(lb_top_df))):
    print(f"\n#{i+1}: {lb_top_df.iloc[i]['model_id']}")
    for col in lb_top_df.columns[1:]:
        print(f"    {col}: {lb_top_df.iloc[i][col]}")

# Performance on test
print("\nBest Model Performance on Test Set (Top Features):")
perf_top = aml_top.leader.model_performance(test)
print(perf_top)

# ============================================================
# 9. Confusion Matrix 
# ============================================================
print("CONFUSION MATRIX - BEST MODEL (ALL FEATURES)")
print("=" * 60)
print(perf_all.confusion_matrix())

# ============================================================
# 10. Summary
# ============================================================
print("SUMMARY: H2O AutoML RESULTS")
print("=" * 60)
print(f"\nAll Features:")
print(f"  Best Model: {lb_df.iloc[0]['model_id']}")
print(f"  Number of models: {len(lb_df)}")
print(f"\nTop 3 Features ({top_3_features}):")
print(f"  Best Model: {lb_top_df.iloc[0]['model_id']}")
print(f"  Number of models: {len(lb_top_df)}")

# Shutdown H2O
h2o.shutdown(prompt=False)
