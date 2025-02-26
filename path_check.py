import os

# Path validation
train_path = "./fine/train"
val_path = "./fine/val"

print(f"Train path exists: {os.path.exists(train_path)}")
print(f"Validation path exists: {os.path.exists(val_path)}")