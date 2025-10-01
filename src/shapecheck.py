import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import get_datasets

train_ds, val_ds, n, classes = get_datasets(img_size=(224,224), batch_size=4)
for x, y in train_ds.take(1):
    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
