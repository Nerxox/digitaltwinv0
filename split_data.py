import os
import pandas as pd

def main():
    # Create datasets directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)
    
    # Check if the input file exists
    input_file = "Cleaned_Production_System_Dataset.csv"
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found in the current directory.")
        print("Available files in the current directory:")
        print("\n".join(f"- {f}" for f in os.listdir('.') if os.path.isfile(f)))
        return

    try:
        # 1. Load the full dataset
        print(f"Loading dataset from {input_file}...")
        df_full = pd.read_csv(input_file)
        
        # 2. Convert 'timestamp' to datetime and set as index
        print("Processing timestamps...")
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        df_full = df_full.set_index('timestamp').sort_index()
        
        # 3. Handle NaNs by interpolation
        print("Handling missing values...")
        df_full = df_full.ffill().bfill()
        
        # 4. Split the data based on time (80% train, 20% test)
        print("Splitting data...")
        split_point = int(len(df_full) * 0.8)
        train_df = df_full.iloc[:split_point]
        test_df = df_full.iloc[split_point:]
        
        # 5. Save the splits
        train_path = "datasets/train1.csv"
        test_path = "datasets/test1.csv"
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        # Print summary
        print("\nData split complete:")
        print(f"- Training data: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"- Testing data:  {len(test_df)} rows, {len(test_df.columns)} columns")
        print(f"\nFiles saved to:\n- {os.path.abspath(train_path)}\n- {os.path.abspath(test_path)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
