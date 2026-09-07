"""Train a simple brain-MRI image classifier.

Expected layout:
data/
  train/
    no_tumor/  *.jpg
    tumor/     *.jpg
  validation/
    no_tumor/  *.jpg
    tumor/     *.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (224, 224)
SEED = 42


def make_model() -> tf.keras.Model:
    """Build a compact transfer-learning classifier."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("model/brain_tumor_classifier.keras"))
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    validation_dir = args.data_dir / "validation"
    for directory in (train_dir, validation_dir):
        if not directory.is_dir():
            raise SystemExit(f"Missing required folder: {directory}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode="int", image_size=IMAGE_SIZE,
        batch_size=args.batch_size, shuffle=True, seed=SEED,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir, label_mode="int", image_size=IMAGE_SIZE,
        batch_size=args.batch_size, shuffle=False,
    )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ])
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=autotune)
    train_ds = train_ds.prefetch(autotune)
    validation_ds = validation_ds.prefetch(autotune)

    model = make_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_auc", mode="max", save_best_only=True),
    ]
    model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, callbacks=callbacks)
    print(f"Saved model to: {args.output}")


if __name__ == "__main__":
    main()
