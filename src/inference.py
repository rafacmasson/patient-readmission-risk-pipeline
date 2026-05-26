import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from typing import Dict, Any, List, Tuple

def load_model_pipeline(model_path: str = "models/readmission_xgb_model.joblib") -> Any:
    """
    Loads the trained model pipeline (including preprocessors and classifier).
    
    Args:
        model_path (str): Path to the saved joblib model pipeline file.
        
    Returns:
        Any: Loaded Scikit-learn Pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
        
    print(f"Loading model pipeline from '{model_path}'...")
    pipeline = joblib.load(model_path)
    return pipeline

def predict_readmission_risk(pipeline: Any, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts the readmission risk probability and class for patients.
    
    Args:
        pipeline (Any): Trained Scikit-learn Pipeline.
        input_data (pd.DataFrame): DataFrame containing the input features.
        
    Returns:
        pd.DataFrame: Original input DataFrame with added 'predicted_label' and 'readmission_probability' columns.
    """
    print("Calculating readmission probabilities...")
    probabilities = pipeline.predict_proba(input_data)[:, 1]
    
    output_df = input_data.copy()
    output_df["readmission_probability"] = probabilities
    return output_df

def calculate_roi_static(
    predictions_df: pd.DataFrame,
    cost_per_readmission: float = 15000.0,
    cost_per_intervention: float = 1200.0,
    intervention_efficacy: float = 0.30,
    probability_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Core ROI calculation helper that computes financial outcomes for a given threshold.
    """
    total_patients = len(predictions_df)
    actual_readmissions = int(predictions_df["label"].sum())
    baseline_readmission_cost = actual_readmissions * cost_per_readmission
    
    # Targeting based on threshold
    targeted_mask = predictions_df["readmission_probability"] >= probability_threshold
    num_targeted = int(targeted_mask.sum())
    total_intervention_cost = num_targeted * cost_per_intervention
    
    # Expected savings (prevented readmissions among targeted * efficacy * cost)
    correctly_targeted_readmissions = int(predictions_df[targeted_mask]["label"].sum())
    expected_prevented_readmissions = correctly_targeted_readmissions * intervention_efficacy
    gross_savings = expected_prevented_readmissions * cost_per_readmission
    
    net_savings = gross_savings - total_intervention_cost
    roi_percentage = (net_savings / total_intervention_cost) * 100 if total_intervention_cost > 0 else 0.0
    
    remaining_readmissions = actual_readmissions - expected_prevented_readmissions
    model_total_cost = total_intervention_cost + (remaining_readmissions * cost_per_readmission)
    
    return {
        "total_patients": total_patients,
        "actual_readmissions": actual_readmissions,
        "patients_targeted": num_targeted,
        "targeting_rate_pct": (num_targeted / total_patients) * 100,
        "baseline_total_cost": baseline_readmission_cost,
        "total_intervention_cost": total_intervention_cost,
        "expected_prevented_readmissions": expected_prevented_readmissions,
        "gross_savings": gross_savings,
        "net_savings": net_savings,
        "roi_pct": roi_percentage,
        "model_total_cost": model_total_cost,
        "cost_reduction_pct": ((baseline_readmission_cost - model_total_cost) / baseline_readmission_cost) * 100 if baseline_readmission_cost > 0 else 0.0
    }

def optimize_threshold_for_savings(
    predictions_df: pd.DataFrame,
    cost_per_readmission: float = 15000.0,
    cost_per_intervention: float = 1200.0,
    intervention_efficacy: float = 0.30,
    save_opt_path: str = "models/threshold_opt_data.joblib"
) -> Tuple[float, Dict[str, Any]]:
    """
    Grid-searches decision thresholds from 0.05 to 0.95 to find the risk threshold 
    that maximizes the net financial savings of the intervention program.
    Saves search data for visualization inside notebooks.
    
    Returns:
        Tuple[float, Dict[str, Any]]: The optimal threshold and its corresponding ROI metrics.
    """
    print("Optimizing decision threshold for maximum financial savings...")
    thresholds = np.arange(0.05, 0.96, 0.05)
    net_savings_list = []
    roi_pct_list = []
    targeted_pct_list = []
    
    best_threshold = 0.50
    best_savings = -float("inf")
    best_metrics = {}
    
    for t in thresholds:
        metrics = calculate_roi_static(
            predictions_df=predictions_df,
            cost_per_readmission=cost_per_readmission,
            cost_per_intervention=cost_per_intervention,
            intervention_efficacy=intervention_efficacy,
            probability_threshold=t
        )
        
        net_savings_list.append(metrics["net_savings"])
        roi_pct_list.append(metrics["roi_pct"])
        targeted_pct_list.append(metrics["targeting_rate_pct"])
        
        if metrics["net_savings"] > best_savings:
            best_savings = metrics["net_savings"]
            best_threshold = t
            best_metrics = metrics
            
    # Save metadata for plotting
    parent_dir = os.path.dirname(save_opt_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        
    print(f"Saving threshold optimization data to '{save_opt_path}'...")
    opt_data = {
        "thresholds": thresholds.tolist(),
        "net_savings": net_savings_list,
        "roi_pct": roi_pct_list,
        "targeted_pct": targeted_pct_list,
        "best_threshold": float(best_threshold),
        "best_savings": float(best_savings)
    }
    joblib.dump(opt_data, save_opt_path)
    
    print(f"Optimization complete. Optimal Threshold found: {best_threshold:.2f} (Savings: ${best_savings:,.2f})")
    return float(best_threshold), best_metrics

def print_roi_report(results: Dict[str, Any], threshold: float) -> None:
    """
    Prints a beautiful executive business report summarizing the ROI.
    """
    print("\n" + "="*50)
    print(f"FINANCIAL ROI ANALYSIS SUMMARY (THRESHOLD: {threshold:.2f})")
    print("="*50)
    print(f"Total Cohort Size:            {results['total_patients']} patients")
    print(f"Baseline Readmissions:        {results['actual_readmissions']}")
    print(f"Baseline Total Cost:          ${results['baseline_total_cost']:,.2f}")
    print("-"*50)
    print(f"Patients Targeted:            {results['patients_targeted']} ({results['targeting_rate_pct']:.1f}%)")
    print(f"Total Intervention Cost:      ${results['total_intervention_cost']:,.2f}")
    print(f"Prevented Readmissions:       {results['expected_prevented_readmissions']:.2f}")
    print(f"Gross Financial Savings:      ${results['gross_savings']:,.2f}")
    print(f"Net Program Savings (Profit): ${results['net_savings']:,.2f}")
    print(f"Return on Investment (ROI):   {results['roi_pct']:.2f}%")
    print(f"New Total System Cost:        ${results['model_total_cost']:,.2f}")
    print(f"Overall Cost Reduction:       {results['cost_reduction_pct']:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    db_path = "healthcare_analytics.db"
    model_path = "models/readmission_xgb_model.joblib"
    
    if os.path.exists(db_path) and os.path.exists(model_path):
        from src.train import load_features_from_db
        df = load_features_from_db(db_path)
        pipeline = load_model_pipeline(model_path)
        predictions_df = predict_readmission_risk(pipeline, df)
        
        # Optimize threshold
        best_t, best_m = optimize_threshold_for_savings(predictions_df)
        print_roi_report(best_m, best_t)
    else:
        print("Database or model file not found. Run main.py, then train.py first.")
