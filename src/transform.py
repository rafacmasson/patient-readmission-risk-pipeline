import pandas as pd
from typing import Tuple

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Loads raw CSV data into a Pandas DataFrame and creates a unique admission identifier.
    
    Args:
        csv_path (str): Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Cleaned raw DataFrame with an added surrogate key 'admission_id'.
    """
    print(f"Loading raw data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    
    # Create a unique identifier for each admission (Surrogate Key)
    df_raw["admission_id"] = df_raw.index + 1
    print(f"Loaded {len(df_raw)} records successfully.")
    return df_raw

def normalize_healthcare_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalizes the flat flat-file dataframe into three distinct relational DataFrames
    and implements clinical-domain feature engineering directly into each schema:
    
    1. Patients (static demographics + 'is_elderly' feature)
    2. Admissions (journey + 'is_high_risk_discharge' feature)
    3. Clinical Details (medical context + comorbidity ratios & prior interaction index)
    
    Args:
        df_raw (pd.DataFrame): Flat raw DataFrame with 'admission_id'.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Normalized dataframes.
    """
    print("Normalizing flat-file schema into relational structures with Feature Engineering...")
    
    # --- Feature Engineering Step 1: Patient-level features ---
    # Create 'is_elderly' feature: Patient age >= 65 (represents high clinical vulnerability)
    df_raw["is_elderly"] = (df_raw["age"] >= 65).astype(int)
    
    # --- Feature Engineering Step 2: Admission-level features ---
    # Create 'is_high_risk_discharge': Discharges to facilities or continuous care
    # (e.g. Nursing Facility, Home Health Care, Rehab) versus home self-care
    high_risk_keywords = "Nursing|Facility|Care|Rehabilitation|Hospice"
    df_raw["is_high_risk_discharge"] = (
        df_raw["discharge_disposition"]
        .str.contains(high_risk_keywords, case=False, na=False)
        .astype(int)
    )
    
    # --- Feature Engineering Step 3: Clinical-level features ---
    # Create 'medication_to_comorbidity_ratio' (avoiding division by zero)
    # High ratio indicates intense medication regimens relative to active diagnoses
    df_raw["medication_to_comorbidity_ratio"] = (
        df_raw["medications_count"] / (df_raw["comorbidities_count"] + 1)
    ).round(4)
    
    # Create 'total_prior_interactions': Sum of prior readmissions and followups
    # (Healthcare system utilization metric)
    df_raw["total_prior_interactions"] = (
        df_raw["prev_readmissions"] + df_raw["followup_visits_last_year"]
    )
    
    # Table 1: Patient Registry (Static demographic data unique per patient)
    patients_df = (
        df_raw[["patient_id", "age", "gender", "region", "is_elderly"]]
        .drop_duplicates(subset=["patient_id"])
    )
    
    # Table 2: Admissions (Journey, administrative data, and the target label)
    admissions_df = df_raw[
        [
            "admission_id",
            "patient_id",
            "admission_date",
            "season",
            "length_of_stay",
            "insurance_type",
            "discharge_disposition",
            "is_high_risk_discharge",
            "label",  # Target Variable: 1 = Readmitted, 0 = Not Readmitted
        ]
    ]
    
    # Table 3: Clinical Details (The medical context of the specific visit)
    clinical_details_df = df_raw[
        [
            "admission_id",
            "primary_diagnosis",
            "treatment_type",
            "comorbidities_count",
            "medications_count",
            "followup_visits_last_year",
            "prev_readmissions",
            "medication_to_comorbidity_ratio",
            "total_prior_interactions"
        ]
    ]
    
    print(f"Schema normalized successfully with 4 new database features:")
    print(f" - Patients Table: {len(patients_df)} unique records (includes 'is_elderly')")
    print(f" - Admissions Table: {len(admissions_df)} records (includes 'is_high_risk_discharge')")
    print(f" - Clinical Details Table: {len(clinical_details_df)} records (includes ratios & utilization indexes)")
    
    return patients_df, admissions_df, clinical_details_df
