
from src.data import get_datasets
train_ds, val_ds, num_classes, class_names = get_datasets(
    images_dir="data/extracted/images/images",
    img_size=(160,160),
    batch_size=8,
    val_split=0.1,
    save_class_map=True
)
print("num_classes:", num_classes)
print("first 5 classes:", class_names[:5])
for images, labels in train_ds.take(1):
    print("batch shape:", images.shape)
    print("labels sample:", labels.numpy()[:5])
    print(class_names[28])