import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils._tags import ClassifierTags

# Patch XGBClassifier to fix compatibility with scikit-learn 1.6+ tag-based system
if hasattr(XGBClassifier, "__sklearn_tags__"):
    original_tags_method = XGBClassifier.__sklearn_tags__
    
    def patched_sklearn_tags(self):
        tags = original_tags_method(self)
        tags.estimator_type = "classifier"
        try:
            if not hasattr(tags, "classifier_tags") or tags.classifier_tags is None:
                tags.classifier_tags = ClassifierTags()
        except Exception:
            pass
        return tags
        
    XGBClassifier.__sklearn_tags__ = patched_sklearn_tags

# Set legacy attribute just in case
XGBClassifier._estimator_type = "classifier"


def load_features_from_db(db_path: str = "healthcare_analytics.db") -> pd.DataFrame:
    """
    Queries the SQLite database to extract demographic, administrative, and clinical features.
    Demonstrates SQL Feature Engineering by computing 'stay_per_comorbidity' inside the query.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Returns:
        pd.DataFrame: Completed modeling dataset.
    """
    print(f"Querying database at '{db_path}' for model features with SQL Feature Engineering...")
    
    # We join all 3 tables and compute stay_per_comorbidity dynamically in SQL
    query = """
        SELECT 
            p.age,
            p.gender,
            p.region,
            p.is_elderly,
            a.season,
            a.length_of_stay,
            a.insurance_type,
            a.discharge_disposition,
            a.is_high_risk_discharge,
            c.primary_diagnosis,
            c.treatment_type,
            c.comorbidities_count,
            c.medications_count,
            c.followup_visits_last_year,
            c.prev_readmissions,
            c.medication_to_comorbidity_ratio,
            c.total_prior_interactions,
            -- SQL Feature Engineering: Length of stay normalized by patient clinical complexity
            (CAST(a.length_of_stay AS REAL) / (c.comorbidities_count + 1.0)) AS stay_per_comorbidity,
            a.label
        FROM admissions a
        JOIN patients p ON a.patient_id = p.patient_id
        JOIN clinical_details c ON a.admission_id = c.admission_id
    """
    connection = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, connection)
        print(f"Successfully loaded {len(df)} records with {df.shape[1] - 1} features.")
        return df
    finally:
        connection.close()

def build_and_train_pipeline(
    df: pd.DataFrame, 
    model_path: str = "models/readmission_xgb_model.joblib",
    eval_data_path: str = "models/evaluation_data.joblib"
) -> Pipeline:
    """
    Preprocesses the dataset, executes a Grid Search Cross-Validation to find the best 
    XGBoost hyperparameters, trains the final pipeline, and saves performance metrics.
    
    Args:
        df (pd.DataFrame): Input modeling dataset.
        model_path (str): Path to save the final trained Pipeline.
        eval_data_path (str): Path to save test data and scores for notebook visualizations.
        
    Returns:
        Pipeline: The optimized Scikit-learn Pipeline.
    """
    # 1. Split features and target
    X = df.drop(columns=["label"])
    y = df["label"]
    
    # 2. Automatically identify categorical and numerical features
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # 3. Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    # 4. Define pipeline scaffolding
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(random_state=42, eval_metric="logloss"))
        ]
    )
    
    # 5. Split train/test sets (stratified to handle class imbalances)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Grid Search Cross-Validation (Hyperparameter Tuning)
    # Define a small, fast, yet practical search space to demonstrate the skill without lagging
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1]
    }
    
    print("Initializing Grid Search CV (5-Fold Cross-Validation)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1
    )
    
    print("Fitting model and searching parameters...")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # 7. Evaluate on held-out test set
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("OPTIMIZED XGBOOST EVALUATION METRICS (TEST SET)")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*50 + "\n")
    
    # 8. Save artifacts (Trained Model & Evaluation Data for plotting)
    for path in [model_path, eval_data_path]:
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
    print(f"Saving trained model pipeline to '{model_path}'...")
    joblib.dump(best_pipeline, model_path)
    
    # Extract feature names after encoding to save for analysis
    cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    try:
        cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    except AttributeError:
        cat_features = cat_encoder.get_feature_names(categorical_cols).tolist()
        
    all_feature_names = cat_features + numerical_cols
    importances = best_pipeline.named_steps["classifier"].feature_importances_
    
    print(f"Saving test predictions and importances to '{eval_data_path}'...")
    eval_data = {
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": all_feature_names,
        "importances": importances.tolist()
    }
    joblib.dump(eval_data, eval_data_path)
    print("Model and evaluation metadata saved successfully.")
    
    return best_pipeline

if __name__ == "__main__":
    db_path = "healthcare_analytics.db"
    if os.path.exists(db_path):
        df = load_features_from_db(db_path)
        build_and_train_pipeline(df)
    else:
        print("Database not found. Run main.py first.")
