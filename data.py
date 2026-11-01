import os
import shutil
import pandas as pd
import kagglehub

DATASET_NAME = "spscientist/students-performance-in-exams"
CSV_FILENAME = "StudentsPerformance.csv"

def load_data(download_dir="."):
    """
    Downloads the student performance dataset from Kaggle if not present locally,
    and returns a pandas DataFrame with standardized column names.
    """
    local_csv = os.path.join(download_dir, CSV_FILENAME)
    
    if not os.path.exists(local_csv):
        print(f"Downloading dataset '{DATASET_NAME}' from Kaggle...")
        cache_path = kagglehub.dataset_download(DATASET_NAME)
        source_csv = os.path.join(cache_path, CSV_FILENAME)
        shutil.copy(source_csv, local_csv)
        print(f"Dataset copied to {local_csv}")
    else:
        print(f"Found local dataset at {local_csv}")

    df = pd.read_csv(local_csv)
    
    # Standardize column names (lowercase and replace spaces/slashes with underscores)
    df.columns = [
        col.strip().lower().replace(" ", "_").replace("/", "_")
        for col in df.columns
    ]
    
    # Add helper columns: total score and average score
    df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
    df["average_score"] = (df["total_score"] / 3).round(2)
    
    return df

if __name__ == "__main__":
    df = load_data()
    print("\nDataset Summary:")
    print(f"Total students: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3))

