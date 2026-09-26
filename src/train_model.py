#!/usr/bin/env python3
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score

def main():
    print("Loading data...")
    # Load the CSV we just generated
    df = pd.read_csv("training_features.csv")
    
    print(f"Loaded {len(df)} candidate pairs.")

    # We group by s1_id to ensure candidates for the same entity stay in the same split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['s1_id']))
    
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    
    features = [
        'name_jaccard', 'addr_jaccard', 'name_contain', 
        'addr_contain', 'name_seq_sim', 'addr_seq_sim', 'country_match'
    ]
    
    X_train, y_train = train_df[features], train_df['label']
    X_val, y_val = val_df[features], val_df['label']
    
    print("Training XGBoost model...")
    # We set scale_pos_weight because true matches (1) are rare compared to false candidates (0)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=25
    )
    
    print("Evaluating thresholds...")
    val_preds = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_f05 = 0
    # Sweeping thresholds to maximize F0.5
    for thresh in [x / 100 for x in range(30, 96, 5)]:
        preds = (val_preds >= thresh).astype(int)
        score = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if score > best_f05:
            best_f05 = score
            best_threshold = thresh
            
    print(f"Best validation F0.5 (pair-level): {best_f05:.4f} at threshold {best_threshold}")
    
    model.save_model("model.xgb")
    print("Model saved to model.xgb")

if __name__ == "__main__":
    main()
