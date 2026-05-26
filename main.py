from src.ingest import download_dataset, locate_raw_csv
from src.transform import load_raw_data, normalize_healthcare_data
from src.database import save_to_database
from src.train import load_features_from_db, build_and_train_pipeline
from src.inference import load_model_pipeline, predict_readmission_risk, optimize_threshold_for_savings, print_roi_report

def run_pipeline() -> None:
    """
    Runs the complete end-to-end machine learning data pipeline:
    1. Downloads raw patient readmission dataset from Kaggle.
    2. Locates the raw CSV file.
    3. Loads and cleans the raw flat dataframe.
    4. Normalizes flat data into three relational schemas (patients, admissions, clinical).
    5. Saves the normalized data tables to an SQLite database.
    6. Queries the database to extract features (with SQL Feature Engineering).
    7. Trains an XGBoost model pipeline with GridSearchCV hyperparameter tuning.
    8. Performs inference (probabilities) on patient readmission risk.
    9. Performs Grid Search Optimization to find the best risk threshold for maximum ROI.
    """
    print("=" * 60)
    print("STARTING COMPLETE PATIENT READMISSION PIPELINE")
    print("=" * 60)
    
    db_path = "healthcare_analytics.db"
    model_path = "models/readmission_xgb_model.joblib"
    
    try:
        # Step 1 & 2: Ingestion
        dataset_dir = download_dataset()
        csv_path = locate_raw_csv(dataset_dir)
        
        # Step 3 & 4: Transformation & Normalization
        df_raw = load_raw_data(csv_path)
        patients_df, admissions_df, clinical_details_df = normalize_healthcare_data(df_raw)
        
        # Step 5: Carga (Load to SQLite)
        save_to_database(
            patients_df=patients_df,
            admissions_df=admissions_df,
            clinical_details_df=clinical_details_df,
            db_path=db_path
        )
        
        # Step 6: Feature Extraction from SQLite (includes stay_per_comorbidity SQL calc)
        df_features = load_features_from_db(db_path=db_path)
        
        # Step 7: Train XGBoost Model Pipeline with Grid Search CV
        pipeline = build_and_train_pipeline(df=df_features, model_path=model_path)
        
        # Step 8: Inference (Risk Probabilities)
        predictions_df = predict_readmission_risk(pipeline=pipeline, input_data=df_features)
        
        # Step 9: Decision Threshold Optimization & Executive Report
        best_t, best_metrics = optimize_threshold_for_savings(predictions_df=predictions_df)
        print_roi_report(results=best_metrics, threshold=best_t)
        
        print("=" * 60)
        print("PIPELINE & MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"PIPELINE FAILED with error: {e}")
        print("=" * 60)
        raise e


if __name__ == "__main__":
    run_pipeline()

