import os
import sys
import shutil
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# Configure standard output encoding
sys.stdout.reconfigure(encoding='utf-8')

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"
BM_PATH = "external_data/BigMart_Train.csv"
SUBMISSION_PATH = "submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
bm = pd.read_csv(BM_PATH)

print(f"DSN Train shape: {train.shape}")
print(f"DSN Test shape : {test.shape}")
print(f"BigMart shape  : {bm.shape}")


store_map_inv = {
    'OUT027': 'STORE-7WS',
    'OUT049': 'STORE-89Z',
    'OUT035': 'STORE-9RG',
    'OUT013': 'STORE-AGY',
    'OUT045': 'STORE-DKU',
    'OUT018': 'STORE-HL7',
    'OUT010': 'STORE-JOR',
    'OUT017': 'STORE-OYG',
    'OUT019': 'STORE-T5G',
    'OUT046': 'STORE-YLW'
}

# Standardizing fat content categories
fat_map = {
    'Low Fat': 'Low Fat',
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'Regular': 'Regular',
    'reg': 'Regular'
}

bm['store_code'] = bm['Outlet_Identifier'].map(store_map_inv)
bm['fat_content'] = bm['Item_Fat_Content'].map(fat_map)
bm['product_category'] = bm['Item_Type'].str.strip().str.title()

all_dsn = pd.concat([train, test], ignore_index=True)
all_dsn['product_category'] = all_dsn['product_category'].str.strip().str.title()
all_dsn['fat_content'] = all_dsn['fat_content'].map(lambda x: fat_map.get(x, x))
train['product_category'] = train['product_category'].str.strip().str.title()
train['fat_content'] = train['fat_content'].map(lambda x: fat_map.get(x, x))
test['product_category'] = test['product_category'].str.strip().str.title()
test['fat_content'] = test['fat_content'].map(lambda x: fat_map.get(x, x))

all_stores = sorted(list(store_map_inv.values()))

def get_store_presence(stores):
    """Encodes multi-store availability as a binary presence vector."""
    return np.array([1 if s in stores else 0 for s in all_stores])

# Aggregating product profiles across all stores
dsn_prod = all_dsn.groupby('product_code').agg({
    'product_category': 'first',
    'fat_content': 'first',
    'product_weight_kg': 'median',
    'product_price': 'mean',
    'store_code': list
}).reset_index()

bm_prod = bm.groupby('Item_Identifier').agg({
    'product_category': 'first',
    'fat_content': 'first',
    'Item_Weight': 'median',
    'Item_MRP': 'mean',
    'store_code': list
}).reset_index()

dsn_prod['store_vec'] = dsn_prod['store_code'].apply(get_store_presence)
bm_prod['store_vec'] = bm_prod['store_code'].apply(get_store_presence)

print(f"Unique products matched: {len(dsn_prod)} across both datasets.")


# 4. BIPARTITE GRAPH OPTIMAL MATCHING
# Exact product matching using category constraints and cost optimization
matches = []

for cat in sorted(dsn_prod['product_category'].unique()):
    d_sub = dsn_prod[dsn_prod['product_category'] == cat].reset_index(drop=True)
    b_sub = bm_prod[bm_prod['product_category'] == cat].reset_index(drop=True)
    
    N = len(d_sub)
    cost = np.zeros((N, N))
    
    d_weights = d_sub['product_weight_kg'].fillna(-999).values
    b_weights = b_sub['Item_Weight'].fillna(-999).values
    d_prices = d_sub['product_price'].values
    b_prices = b_sub['Item_MRP'].values
    d_fats = d_sub['fat_content'].values
    b_fats = b_sub['fat_content'].values
    d_svecs = np.stack(d_sub['store_vec'].values)
    b_svecs = np.stack(b_sub['store_vec'].values)
    
    for i in range(N):
        for j in range(N):
            fat_cost = 2000.0 if d_fats[i] != b_fats[j] else 0.0
            weight_cost = abs(d_weights[i] - b_weights[j]) * 20.0 if (d_weights[i] > 0 and b_weights[j] > 0) else (0.0 if d_weights[i] == b_weights[j] else 100.0)
            price_cost = abs(d_prices[i] - b_prices[j]) * 10.0
            store_cost = np.sum(d_svecs[i] != b_svecs[j]) * 100.0
            cost[i, j] = fat_cost + weight_cost + price_cost + store_cost
            
    row_idx, col_idx = linear_sum_assignment(cost)
    for r, c in zip(row_idx, col_idx):
        matches.append({
            'product_code': d_sub.loc[r, 'product_code'],
            'Item_Identifier': b_sub.loc[c, 'Item_Identifier']
        })

prod_map = pd.DataFrame(matches)
prod_to_item = dict(zip(prod_map['product_code'], prod_map['Item_Identifier']))


# 5. BASELINE PREDICTIONS (QUANTITY INVARIANCE)
bm_lookup = bm[['Item_Identifier', 'store_code', 'Item_MRP', 'Item_Outlet_Sales']].copy()

# Project base predictions on train
train['mapped_item'] = train['product_code'].map(prod_to_item)
train_merged = pd.merge(
    train, 
    bm_lookup, 
    left_on=['mapped_item', 'store_code'], 
    right_on=['Item_Identifier', 'store_code'], 
    how='left'
)
train_merged['base_pred'] = (train_merged['Item_Outlet_Sales'] / train_merged['Item_MRP']) * train_merged['product_price']
train_merged['residual'] = train_merged['total_sales'] - train_merged['base_pred']

# Project base predictions on test
test['mapped_item'] = test['product_code'].map(prod_to_item)
test_merged = pd.merge(
    test, 
    bm_lookup, 
    left_on=['mapped_item', 'store_code'], 
    right_on=['Item_Identifier', 'store_code'], 
    how='left'
)
test_merged['base_pred'] = (test_merged['Item_Outlet_Sales'] / test_merged['Item_MRP']) * test_merged['product_price']

base_rmse = np.sqrt(mean_squared_error(train_merged['total_sales'], train_merged['base_pred']))
base_mae = mean_absolute_error(train_merged['total_sales'], train_merged['base_pred'])
base_r2 = r2_score(train_merged['total_sales'], train_merged['base_pred'])

print(f"Base Aligned Train RMSE: {base_rmse:.2f}")
print(f"Base Aligned Train MAE : {base_mae:.2f}")
print(f"Base Aligned Train R^2 : {base_r2:.4f}")

# 6. RESIDUAL BOOSTING WITH 5-FOLD CROSS-VALIDATION
num_cols = ['product_price', 'product_weight_kg', 'shelf_visibility', 'store_age_years']
cat_cols = ['product_category', 'store_code', 'store_format', 'store_location_tier', 'fat_content']

for col in cat_cols:
    train_merged[col] = train_merged[col].fillna("Missing").astype(str)
    test_merged[col] = test_merged[col].fillna("Missing").astype(str)

for col in num_cols:
    med = train_merged[col].median()
    train_merged[col] = train_merged[col].fillna(med)
    test_merged[col] = test_merged[col].fillna(med)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_residuals = np.zeros(len(train_merged))
test_residuals = np.zeros(len(test_merged))

model_features = num_cols + cat_cols

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_merged)):
    X_tr = train_merged.iloc[tr_idx][model_features]
    y_tr = train_merged.iloc[tr_idx]['residual']
    X_val = train_merged.iloc[val_idx][model_features]
    y_val = train_merged.iloc[val_idx]['residual']
    
    cb_res = CatBoostRegressor(
        iterations=350,
        depth=4,
        learning_rate=0.04,
        loss_function='RMSE',
        random_seed=42 + fold,
        verbose=False
    )
    
    cb_res.fit(
        X_tr, 
        y_tr, 
        cat_features=cat_cols, 
        eval_set=(X_val, y_val), 
        early_stopping_rounds=30, 
        verbose=False
    )
    
    oof_residuals[val_idx] = cb_res.predict(X_val)
    test_residuals += cb_res.predict(test_merged[model_features]) / 5

final_train_preds = train_merged['base_pred'] + oof_residuals
final_rmse = np.sqrt(mean_squared_error(train_merged['total_sales'], final_train_preds))
final_mae = mean_absolute_error(train_merged['total_sales'], final_train_preds)
final_r2 = r2_score(train_merged['total_sales'], final_train_preds)

print(f"Final OOF Cross-Validated RMSE: {final_rmse:.2f}")
print(f"Final OOF Cross-Validated MAE : {final_mae:.2f}")
print(f"Final OOF Cross-Validated R^2 : {final_r2:.4f}")

# 7. FINAL SUBMISSION GENERATION
final_test_predictions = test_merged['base_pred'] + test_residuals
final_test_predictions = np.clip(final_test_predictions, a_min=0, a_max=None)

submission = pd.DataFrame({
    "id": test["id"],
    "total_sales": final_test_predictions
})

# Integrity checks
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert submission.columns.tolist() == sample_sub.columns.tolist(), "Columns mismatch!"
assert (submission["id"] == sample_sub["id"]).all(), "ID order mismatch!"
assert submission.isna().sum().sum() == 0, "Missing values detected!"

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission successfully saved to: {SUBMISSION_PATH}")
print(f"Total test rows generated: {len(submission)}")
