import wandb
import pandas as pd
import json
import os

# 1. Initialize API
api = wandb.Api()
runs = api.runs("bearbug/Colorful_Cutout_Aug")

results = []

print(f"Found {len(runs)} runs. Extracting Acc and Data_fraction...")

for run in runs:
    # 2. Find the artifact
    artifacts = run.logged_artifacts()
    
    for artifact in artifacts:
        if "TEST_Result" in artifact.name:
            try:
                # 3. Download and read the table
                dir_path = artifact.download()
                file_path = os.path.join(dir_path, "TEST_Result.table.json")
                
                with open(file_path, 'r') as f:
                    table_dict = json.load(f)
                
                # 4. Convert to DataFrame
                df = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
                
                # 5. Extract specific columns (Acc and Data_fraction)
                # We use .iloc[0] assuming there is one result row per run
                results.append({
                    "Run Name": run.name,
                    "Accuracy": df["Acc"].iloc[0],
                    # Extracting exactly as it appears in your screenshot column header
                    "Data Fraction": df["Data_fraction"].iloc[0] 
                })
                
                break 
                
            except Exception as e:
                print(f"Error extracting from {run.name}: {e}")

# 6. Create the final clean table
final_df = pd.DataFrame(results)

# Sort by Run Name for cleaner viewing
final_df = final_df.sort_values("Run Name")

print("-" * 30)
print(final_df)

# Save to CSV
final_df.to_csv("wandb_acc_datafrac.csv", index=False)
print("Saved to wandb_acc_datafrac.csv")