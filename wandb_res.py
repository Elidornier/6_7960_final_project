import wandb
import pandas as pd
import json
import os

# 1. Initialize API
api = wandb.Api()
# Based on your screenshot, your entity is 'bearbug' and project is 'Colorful_Cutout_Aug'
runs = api.runs("bearbug/Colorful_Cutout_Aug")

results = []

print(f"Found {len(runs)} runs. Extracting data...")

for run in runs:
    # 2. Find the specific artifact containing your table
    # W&B stores tables as artifacts. We iterate through them to find "TEST_Result"
    artifacts = run.logged_artifacts()
    
    for artifact in artifacts:
        # We look for the artifact that matches your table name
        if "TEST_Result" in artifact.name:
            try:
                # 3. Download the table file
                # This downloads the JSON file representing the table to a local folder
                dir_path = artifact.download()
                
                # The file is typically named "TEST_Result.table.json"
                file_path = os.path.join(dir_path, "TEST_Result.table.json")
                
                with open(file_path, 'r') as f:
                    table_dict = json.load(f)
                
                # 4. Convert JSON to Pandas DataFrame
                df = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
                
                # 5. Extract the specific "Acc" value
                # Using .iloc[0] because your screenshot shows only 1 row per table
                acc_value = df["Acc"].iloc[0]
                
                results.append({
                    "Run Name": run.name,
                    "Accuracy": acc_value
                })
                
                # Break the inner loop once found for this run
                break 
                
            except Exception as e:
                print(f"Could not extract from {run.name}: {e}")

# 6. Create the final clean table
final_df = pd.DataFrame(results)
print("-" * 30)
print(final_df)

# Optional: Save to CSV
final_df.to_csv("extracted_accuracies.csv", index=False)