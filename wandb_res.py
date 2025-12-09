import wandb
import pandas as pd
import json
import os

# 1. Initialize API
api = wandb.Api()
# Replace with your actual entity/project
runs = api.runs("bearbug/Colorful_Cutout_Aug")

results = []

print(f"Found {len(runs)} total runs. Filtering for ' - Test' runs...")

for run in runs:
    # --- FILTERING STEP ---
    # Only keep runs that have " - Test" in the name
    if " - Test" not in run.name:
        continue
    # ----------------------

    # Initialize defaults (Empty by default)
    run_acc = None 
    run_frac = None
    
    # 2. Try to find the data in the artifact
    try:
        for artifact in run.logged_artifacts():
            if "TEST_Result" in artifact.name:
                # Download and read
                dir_path = artifact.download()
                file_path = os.path.join(dir_path, "TEST_Result.table.json")
                
                with open(file_path, 'r') as f:
                    table_dict = json.load(f)
                
                df_table = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
                
                # Extract Acc
                if "Acc" in df_table.columns:
                    run_acc = df_table["Acc"].iloc[0]
                
                # Extract Data_fraction if it exists, otherwise leave as None
                if "Data_fraction" in df_table.columns:
                    run_frac = df_table["Data_fraction"].iloc[0]
                
                # Stop looking through artifacts for this run once we found the table
                break 
    except Exception as e:
        # If extraction fails, we print a warning but still keep the run in the list (as None)
        # This ensures runs without valid tables/datafracs are still listed if they match the name.
        pass

    # 3. Append the filtered run
    results.append({
        "Run Name": run.name,
        "Accuracy": run_acc,
        "Data Fraction": run_frac
    })

# 4. Create and Save
final_df = pd.DataFrame(results)
final_df = final_df.sort_values("Run Name")

print("-" * 30)
print(f"Kept {len(final_df)} runs after filtering.")
print(final_df.head())

final_df.to_csv("wandb_test_runs_only.csv", index=False)
print("Saved to wandb_test_runs_only.csv")