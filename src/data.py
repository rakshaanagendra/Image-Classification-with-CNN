import os
import json
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def _get_image_files_and_labels(images_dir):
    """
    Scan images_dir and return lists:
      - file_paths (strings)
      - labels (ints)
      - class_names (sorted list of class name strings)

    For filenames like 'american_bulldog_12.jpg' we take 'american_bulldog' as the class.
    """
    exts = (".jpg", ".jpeg", ".png")
    file_paths = []
    class_names = []
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(exts):
            continue
        file_paths.append(os.path.join(images_dir, fname))
        base = os.path.splitext(fname)[0]   # "american_bulldog_12"
        parts = base.split("_")
        if len(parts) >= 2:
            cls = "_".join(parts[:-1])
        else:
            cls = parts[0]
        class_names.append(cls)
    unique_names = sorted(set(class_names))
    name_to_idx = {n: i for i, n in enumerate(unique_names)}
    labels = [name_to_idx[c] for c in class_names]
    return file_paths, labels, unique_names


def _load_and_preprocess(path, label, img_size=(224,224)):
    """
    path: tf.string scalar tensor containing file path
    label: tf.int32 scalar tensor
    Returns (image_tensor, label_tensor)
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)   # force RGB
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0        # normalize to [0,1]
    return image, label


def prepare_for_training(ds, batch_size=32, shuffle_buffer=1000, augment_fn=None):
    ds = ds.shuffle(shuffle_buffer, seed=123)
    if augment_fn:
        ds = ds.map(lambda x, y: (augment_fn(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_datasets(images_dir="data/extracted/images/images", img_size=(224,224),
                 batch_size=32, val_split=0.15, seed=123, save_class_map=True):
    """
    Loads local images and returns train/val datasets and class names.
    """
    images_dir = os.path.abspath(images_dir)
    file_paths, labels, class_names = _get_image_files_and_labels(images_dir)

    if len(file_paths) == 0:
        raise RuntimeError(f"No image files found in {images_dir}")

    file_paths = np.array(file_paths, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(file_paths))
    rng.shuffle(idx)
    file_paths, labels = file_paths[idx], labels[idx]

    n_val = int(len(file_paths) * val_split)
    val_paths, val_labels = file_paths[:n_val], labels[:n_val]
    train_paths, train_labels = file_paths[n_val:], labels[n_val:]

    if save_class_map:
        data_dir = os.path.dirname(images_dir)
        cm_path = os.path.abspath(os.path.join(data_dir, "class_map.json"))
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_ds = tf.data.Dataset.from_tensor_slices((list(train_paths), list(train_labels)))
    val_ds = tf.data.Dataset.from_tensor_slices((list(val_paths), list(val_labels)))

    train_ds = train_ds.map(lambda p, l: _load_and_preprocess(p, l, img_size),
                            num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda p, l: _load_and_preprocess(p, l, img_size),
                        num_parallel_calls=AUTOTUNE)

    train_ds = prepare_for_training(train_ds, batch_size=batch_size)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, len(class_names), class_names
