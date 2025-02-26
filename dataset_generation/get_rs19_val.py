import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("ngohoang0207/railsem19")
os.mkdir("rs19_val", exist_ok=True)

os.system(f"mv {path}/* rs19_val")
