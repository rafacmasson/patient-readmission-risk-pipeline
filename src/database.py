import sqlite3
import os
import pandas as pd

def save_to_database(
    patients_df: pd.DataFrame, 
    admissions_df: pd.DataFrame, 
    clinical_details_df: pd.DataFrame, 
    db_path: str = "healthcare_analytics.db"
) -> None:
    """
    Establishes a connection to an SQLite database and saves the normalized DataFrames 
    to their respective tables.
    
    Args:
        patients_df (pd.DataFrame): Demographic data.
        admissions_df (pd.DataFrame): Admission & administrative records.
        clinical_details_df (pd.DataFrame): Clinical detail records.
        db_path (str): Path to the target SQLite database file. Defaults to 'healthcare_analytics.db'.
    """
    # Ensure parent directories exist (useful if storing in data/processed/)
    parent_dir = os.path.dirname(db_path)
    if parent_dir and not os.path.exists(parent_dir):
        print(f"Creating directory: {parent_dir}")
        os.makedirs(parent_dir, exist_ok=True)
        
    print(f"Connecting to database at '{db_path}'...")
    connection = sqlite3.connect(db_path)
    
    try:
        # Load tables
        print("Saving 'patients' table...")
        patients_df.to_sql("patients", connection, if_exists="replace", index=False)
        
        print("Saving 'admissions' table...")
        admissions_df.to_sql("admissions", connection, if_exists="replace", index=False)
        
        print("Saving 'clinical_details' table...")
        clinical_details_df.to_sql("clinical_details", connection, if_exists="replace", index=False)
        
        # Verification query to check loaded tables
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Database schema successfully created with tables: {tables}")
        
    except Exception as e:
        print(f"An error occurred while loading data into database: {e}")
        raise e
        
    finally:
        connection.close()
        print("Database connection closed.")
