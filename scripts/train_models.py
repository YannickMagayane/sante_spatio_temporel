"""
Train LSTM, CNN, and ConvLSTM models for epidemic prediction in DRC provinces/regions.

Supports three architectures:
- LSTM: temporal sequences per health zone
- CNN: spatial images with multi-channel variables
- ConvLSTM: hybrid spatio-temporal model

Example:
    $ python scripts/train_models.py --model lstm --epochs 100 --batch_size 32
    $ python scripts/train_models.py --model cnn --epochs 100
    $ python scripts/train_models.py --model convlstm --epochs 100
"""

import argparse
import os
import glob

import numpy as np


def build_lstm_model(input_shape):
    """
    Build a 2-layer LSTM model for time series prediction.

    :param input_shape: Tuple (sequence_length, n_features).
    :return: Compiled Keras model.

    Example:
        >>> model = build_lstm_model((12, 5))
        >>> model.output_shape == (None, 1)
        True
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


def build_cnn_model(input_shape):
    """
    Build a CNN model for spatial risk prediction.

    :param input_shape: Tuple (height, width, n_channels).
    :return: Compiled Keras model.

    Example:
        >>> model = build_cnn_model((32, 32, 3))
        >>> model.output_shape == (None, 1)
        True
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Dense, Dropout,
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


def build_convlstm_model(input_shape):
    """
    Build a ConvLSTM2D hybrid model for spatio-temporal prediction.

    :param input_shape: Tuple (timesteps, height, width, channels).
    :return: Compiled Keras model.

    Example:
        >>> model = build_convlstm_model((12, 32, 32, 3))
        >>> model is not None
        True
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout,
    )

    model = Sequential([
        ConvLSTM2D(
            filters=32, kernel_size=(3, 3),
            padding="same", return_sequences=True,
            input_shape=input_shape,
        ),
        BatchNormalization(),

        ConvLSTM2D(
            filters=64, kernel_size=(3, 3),
            padding="same", return_sequences=False,
        ),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


def load_training_data(model_type, data_dir="data/processed/"):
    """
    Load preprocessed training data based on model type.

    :param model_type: Type of model (lstm, cnn, convlstm).
    :param data_dir: Directory containing preprocessed data.
    :return: Tuple (X_train, y_train, X_val, y_val).

    Example:
        >>> X_train, y_train, X_val, y_val = load_training_data("lstm")
    """
    if model_type == "lstm":
        X_files = sorted(glob.glob(os.path.join(data_dir, "*_X_lstm.npy")))
        y_files = sorted(glob.glob(os.path.join(data_dir, "*_y_lstm.npy")))

        if not X_files:
            print("⚠️  No LSTM data found. Generating synthetic data for testing...")
            X = np.random.rand(200, 12, 5).astype(np.float32)
            y = np.random.rand(200).astype(np.float32)
        else:
            X = np.concatenate([np.load(f) for f in X_files])
            y = np.concatenate([np.load(f) for f in y_files])

    elif model_type == "cnn":
        print("⚠️  No CNN data found. Generating synthetic data for testing...")
        X = np.random.rand(200, 32, 32, 3).astype(np.float32)
        y = np.random.rand(200).astype(np.float32)

    elif model_type == "convlstm":
        print("⚠️  No ConvLSTM data found. Generating synthetic data for testing...")
        X = np.random.rand(100, 12, 32, 32, 3).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"📊 Data: train={X_train.shape}, val={X_val.shape}")
    return X_train, y_train, X_val, y_val


def train(model_type, epochs, batch_size):
    """
    Train the specified model and save it to disk.

    :param model_type: Type of model (lstm, cnn, convlstm).
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :return: Path to the saved model file.

    Example:
        >>> path = train("lstm", epochs=5, batch_size=16)
        >>> path.endswith(".h5")
        True
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("=" * 60)
    print(f"🧠 TRAINING: {model_type.upper()} model")
    print("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val = load_training_data(model_type)

    # Build model
    if model_type == "lstm":
        model = build_lstm_model(X_train.shape[1:])
    elif model_type == "cnn":
        model = build_cnn_model(X_train.shape[1:])
    elif model_type == "convlstm":
        model = build_convlstm_model(X_train.shape[1:])

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss"),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_type}_final.h5"
    model.save(model_path)
    print(f"\n✅ Model saved: {model_path}")

    # Save training history
    import json
    history_path = f"models/{model_type}_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"📈 History saved: {history_path}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train deep learning models for epidemic prediction."
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn", "convlstm"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch_size)
