import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Housing.csv")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Target and Features
    y = np.log1p(df["price"])
    X = df.drop(columns=["price"])
    
    num_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    cat_cols = [
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea", "furnishingstatus"
    ]
    
    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    
    # Ridge regression with alpha=10
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", Ridge(alpha=10.0))
        ]
    )
    
    print("Fitting Ridge pipeline on full dataset...")
    pipeline.fit(X, y)
    
    feature_order = X.columns.tolist()
    
    # Targets for export
    export_targets = [
        base_dir,
        os.path.join(base_dir, "Project")
    ]
    
    for target in export_targets:
        os.makedirs(target, exist_ok=True)
        model_path = os.path.join(target, "house_price_model.pkl")
        features_path = os.path.join(target, "model_features.pkl")
        
        joblib.dump(pipeline, model_path)
        joblib.dump(feature_order, features_path)
        print(f"✅ Saved model & features to: {target}")

    # Sanity test
    test_sample = pd.DataFrame([{
        "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
        "mainroad": "yes", "guestroom": "no", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "yes", "parking": 2,
        "prefarea": "yes", "furnishingstatus": "furnished"
    }])[feature_order]
    
    pred_log = pipeline.predict(test_sample)
    pred_price = np.expm1(pred_log[0])
    print(f"Sanity Check Prediction: ₹{round(pred_price, 2):,} (Expected ~₹10M-₹13M)")

if __name__ == "__main__":
    main()
