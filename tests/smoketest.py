# tests/smoketest.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from src.data import get_datasets
from src.model import build_model
import pytest

# ensure small test dataset exists (will be called in CI)
from tests.generate_test_data import ensure_test_data
ensure_test_data(root="tests/data", img_size=(64,64))

TEST_IMAGES_DIR = "tests/data"

def test_data_loading():
    train_ds, val_ds, num_classes, class_names = get_datasets(
        images_dir=TEST_IMAGES_DIR,
        img_size=(64, 64),
        batch_size=2,
        val_split=0.5,
        seed=123,
        save_class_map=False
    )
    batch_x, batch_y = next(iter(train_ds))
    assert batch_x.shape[-1] == 3   # RGB
    assert batch_y.shape[0] == 2
    assert num_classes >= 2

def test_model_build():
    # Build the model (small input) - this will attempt to load weights (may download),
    # but it's acceptable for CI; if you want to skip weight download, see note below.
    model = build_model(num_classes=2, img_size=(64,64))
    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape[1] == 2

def test_one_training_step():
    train_ds, val_ds, num_classes, class_names = get_datasets(
        images_dir=TEST_IMAGES_DIR,
        img_size=(64, 64),
        batch_size=2,
        val_split=0.5,
        seed=123,
        save_class_map=False
    )
    model = build_model(num_classes=num_classes, img_size=(64,64))
    # run a single training epoch on tiny batch to validate end-to-end
    history = model.fit(train_ds.take(1), validation_data=val_ds.take(1), epochs=1, verbose=0)
    assert "loss" in history.history
