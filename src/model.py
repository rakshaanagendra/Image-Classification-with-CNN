import tensorflow as tf
from typing import Tuple

def build_model(
    num_classes: int,
    img_size: Tuple[int, int] = (224, 224),
    base_trainable: bool = False,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    # 1) Input layer (force RGB 3 channels)
    inputs = tf.keras.Input(shape=(*img_size, 3), name="input_image")

    # 2) Data augmentation
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
    ], name="data_augmentation")(inputs)

    # 3) Rescale back to [0,255] → EfficientNet expects raw ImageNet RGB
    x = tf.keras.layers.Rescaling(255.0)(x)

    # 4) Build EfficientNetB0 WITHOUT weights first (force correct input shape)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,                     # ⛔ don't load yet
        input_shape=(*img_size, 3)        # ✅ enforce RGB
    )

    # 5) Manually load official ImageNet weights (no top)
    weights_path = tf.keras.utils.get_file(
        "efficientnetb0_notop.h5",
        "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
    )
    base_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    base_model.trainable = base_trainable

    # Apply features
    x = base_model(x, training=False)

    # 6) Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    # 7) Build & compile
    model = tf.keras.Model(inputs, outputs, name="efficientnet_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int) -> None:
    """Unfreeze the top `n_layers` layers for fine-tuning."""
    all_layers = []

    def _collect(layer):
        if isinstance(layer, tf.keras.Model):
            for l in layer.layers:
                _collect(l)
        else:
            all_layers.append(layer)

    for layer in model.layers:
        _collect(layer)

    for layer in all_layers:
        layer.trainable = False
    for layer in all_layers[-n_layers:]:
        layer.trainable = True


if __name__ == "__main__":
    m = build_model(num_classes=5, img_size=(160, 160))
    print("EfficientNet input shape:", m.input_shape)
    m.summary()
