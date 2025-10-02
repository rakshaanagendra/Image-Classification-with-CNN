import os
import tensorflow as tf
from src.data import get_datasets
from src.model import build_model
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_data_loading():
    """Check if data pipeline returns a non-empty batch with correct shapes."""
    train_ds, val_ds, num_classes, class_names = get_datasets(
        images_dir="data/extracted/images/images",
        img_size=(64, 64),   # smaller size for fast CI
        batch_size=2,
        val_split=0.1,
        seed=123,
        save_class_map=False
    )

    batch_x, batch_y = next(iter(train_ds))
    assert batch_x.shape[-1] == 3   # RGB
    assert batch_y.shape[0] == 2    # batch size
    assert num_classes > 1          # should detect multiple classes

def test_model_build():
    """Check if model builds successfully and outputs correct shape."""
    num_classes = 5
    model = build_model(num_classes=num_classes, img_size=(64, 64))
    # check input/output shape
    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape == (None, num_classes)

def test_one_training_step():
    """Run one training step to confirm model + data are compatible."""
    train_ds, val_ds, num_classes, class_names = get_datasets(
        images_dir="data/extracted/images/images",
        img_size=(64, 64),
        batch_size=2,
        val_split=0.1,
        seed=123,
        save_class_map=False
    )

    model = build_model(num_classes=num_classes, img_size=(64, 64))
    history = model.fit(train_ds.take(1), validation_data=val_ds.take(1), epochs=1)
    # ensure training history contains loss
    assert "loss" in history.history
