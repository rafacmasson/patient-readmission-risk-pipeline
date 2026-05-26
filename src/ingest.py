import os
import kagglehub

def download_dataset() -> str:
    """
    Downloads the hospital patient readmission dataset from Kaggle using kagglehub.
    
    Returns:
        str: Absolute path to the downloaded dataset directory.
    """
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mohamedasak/hospital-patient-readmission-dataset")
    print(f"Dataset downloaded successfully to: {path}")
    return path

def locate_raw_csv(dataset_dir: str) -> str:
    """
    Locates the raw CSV file in the downloaded dataset directory.
    
    Args:
        dataset_dir (str): Absolute path to the downloaded dataset directory.
        
    Returns:
        str: Absolute path to the raw CSV file.
        
    Raises:
        FileNotFoundError: If no CSV file is found in the directory.
    """
    files = os.listdir(dataset_dir)
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in dataset directory: {dataset_dir}")
    
    csv_path = os.path.join(dataset_dir, csv_files[0])
    print(f"Raw CSV file located: {csv_path}")
    return csv_path
