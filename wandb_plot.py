import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
ENTITY = "bearbug"              
PROJECT = "Colorful_Cutout_Aug" 

# YOUR CORRECT KEYS:
METRIC_TRAIN = "TRAIN/Epoch_Acc"
METRIC_VAL   = "VALID/Epoch_Acc"
X_AXIS       = "Epoch_Index"      

# ==========================================
# 2. FETCH DATA
# ==========================================
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

cifar10_data = []
cifar100_data = []

print(f"Scanning {len(runs)} total runs...")

for run in runs:
    # --- FILTER 1: Exclude Test Runs ---
    if " - Test" in run.name:
        continue

    # --- FILTER 2: Exclude Data Fraction = 1.0 ---
    df_conf = run.config.get('data_fraction')
    if df_conf == 1.0 or df_conf == '1.0':
        continue

    # --- FILTER 3: Identify Dataset ---
    dataset = run.config.get('dataset', '').lower()
    if not dataset:
        if 'cifar100' in run.name.lower():
            dataset = 'cifar100'
        elif 'cifar10' in run.name.lower():
            dataset = 'cifar10'
    
    if dataset not in ['cifar10', 'cifar100']:
        continue 

    # --- FETCH HISTORY ---
    try:
        hist = run.history(keys=[X_AXIS, METRIC_TRAIN, METRIC_VAL])
        
        if hist.empty:
            continue
            
        clean_name = run.name.split(' / ')[-1] if ' /' in run.name else run.name
        
        hist['Method'] = clean_name
        hist['Run Name'] = run.name
        
        if dataset == 'cifar10':
            cifar10_data.append(hist)
        else:
            cifar100_data.append(hist)
            
    except Exception as e:
        print(f"Error fetching {run.name}: {e}")

print(f"\nFound {len(cifar10_data)} valid runs for CIFAR-10")
print(f"Found {len(cifar100_data)} valid runs for CIFAR-100")

# ==========================================
# 3. PLOTTING FUNCTION (UPDATED)
# ==========================================
def plot_graph(data_list, title, y_metric, filename):
    if not data_list:
        print(f"Skipping {title} (No data found)")
        return

    df = pd.concat(data_list)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # --- MODIFIED PLOTTING CALL ---
    sns.lineplot(
        data=df, 
        x=X_AXIS, 
        y=y_metric, 
        hue="Method", 
        style="Method",   # Gives different markers (circle, X, square) per method
        markers=True,     # ENABLE MARKERS
        dashes=False,     
        markersize=8,     # Make them large and visible
        markeredgecolor="white", # Add white border to make them pop
        linewidth=2.0,
        alpha=0.9
    )
    # ------------------------------

    plt.title(title, fontsize=15, weight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    
    # Legend formatting
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Augmentation Method", frameon=False)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.show()

# ==========================================
# 4. GENERATE PLOTS
# ==========================================
print("\nGenerating Plots with Dots...")

# CIFAR 10
plot_graph(cifar10_data, "CIFAR-10: Training Accuracy", METRIC_TRAIN, "c10_train_dots.png")
plot_graph(cifar10_data, "CIFAR-10: Validation Accuracy", METRIC_VAL,   "c10_val_dots.png")

# CIFAR 100
plot_graph(cifar100_data, "CIFAR-100: Training Accuracy", METRIC_TRAIN, "c100_train_dots.png")
plot_graph(cifar100_data, "CIFAR-100: Validation Accuracy", METRIC_VAL,   "c100_val_dots.png")