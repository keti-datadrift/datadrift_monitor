import kagglehub

# Download latest version
path = kagglehub.dataset_download("renancostaalencar/compcars")

print("Path to dataset files:", path)