import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import tensorflow as tf
from src.data import get_datasets
from src.model import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size

    # 1) Load datasets
    train_ds, val_ds, num_classes, class_names = get_datasets(
        images_dir="data/extracted/images/images",
        img_size=img_size,
        batch_size=batch_size,
        val_split=0.15,
        seed=123,
        save_class_map=True
    )

    # 2) Build model
    model = build_model(num_classes=num_classes, img_size=img_size, base_trainable=not args.freeze_base)

    # 3) Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    # 4) Train
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 5) Save final model
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/pet_classifier.keras")  # ✅ new Keras format
    print("Training finished. Model saved to saved_models/pet_classifier.keras")

if __name__ == "__main__":
    main()
