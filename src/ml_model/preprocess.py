import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_dataset():
    raw_path = "dataset/raw/final_research_dataset_MASTER.csv"
    output_dir = "dataset/processed"
    output_path = os.path.join(output_dir, "clean_builds.csv")
    
    print("--- Step 1: Loading Raw CSV ---")
    df = pd.read_csv(raw_path)
    print(f"Initial Shape: {df.shape}")
    
    print("\n--- Step 2: Filtering 'conclusion' Column ---")
    print("Original value counts:\n", df['conclusion'].value_counts(dropna=False))
    df = df[df['conclusion'].isin(['success', 'failure'])]
    print(f"Shape after filtering success/failure: {df.shape}")
    
    print("\n--- Step 3: Creating Binary Label 'build_failed' ---")
    # 1 = failure, 0 = success
    df['build_failed'] = df['conclusion'].apply(lambda x: 1 if x == 'failure' else 0)
    print("Label balance:\n", df['build_failed'].value_counts())
    
    print("\n--- Step 4: Dropping Nulls in Target Features ---")
    feature_cols = [
        'duration', 'additions', 'deletions', 'total_churn', 'files_modified', 
        'msg_len', 'is_merge', 'num_parents', 'run_attempt', 
        'time_since_last_commit', 'event', 'head_branch'
    ]
    # Ensure all required features exist in dataframe before dropping nulls
    existing_features = [col for col in feature_cols if col in df.columns]
    missing_features = set(feature_cols) - set(existing_features)
    if missing_features:
        print(f"Warning: Missing columns from dataset: {missing_features}")
        
    df = df.dropna(subset=existing_features)
    print(f"Shape after dropping nulls from features: {df.shape}")
    
    print("\n--- Step 5: Dropping Exact Duplicate Rows ---")
    df = df.drop_duplicates()
    print(f"Shape after dropping duplicates: {df.shape}")
    
    print("\n--- Step 6: Stratified Sampling to 40k Rows ---")
    target_sample_size = 40000
    if len(df) > target_sample_size:
        # Use train_test_split stratify to maintain target ratio perfectly
        df, _ = train_test_split(
            df, 
            train_size=target_sample_size, 
            stratify=df['build_failed'], 
            random_state=42
        )
    print(f"Final sampled shape: {df.shape}")
    print("Final label balance:\n", df['build_failed'].value_counts(normalize=True))
    
    print("\n--- Step 7: Saving Processed File ---")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved to: {output_path}")

if __name__ == "__main__":
    preprocess_dataset()
