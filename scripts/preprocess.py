"""
Preprocess climate, sanitary, and scraped data for model training.

Handles missing values, temporal aggregation, sequence creation for LSTM,
and spatial image construction for CNN.

Example:
    $ python scripts/preprocess.py --config config/preprocess.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml


def load_config(config_path):
    """
    Load preprocessing configuration from YAML file.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary of configuration parameters.

    Example:
        >>> config = load_config("config/preprocess.yaml")
        >>> "cities" in config
        True
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_climate_csv(climate_dir, cities):
    """
    Load climate CSV files for all cities into a single DataFrame.

    :param climate_dir: Directory containing climate CSVs.
    :param cities: List of city names.
    :return: pandas DataFrame with columns [date, city, temperature, precipitation].

    Example:
        >>> df = load_climate_csv("data/climat/", ["kinshasa"])
        >>> "temperature" in df.columns
        True
    """
    frames = []
    for city in cities:
        pattern = os.path.join(climate_dir, f"{city}_*.csv")
        import glob
        files = sorted(glob.glob(pattern))
        for f in files:
            try:
                df = pd.read_csv(f, parse_dates=["date"])
                df["city"] = city
                frames.append(df)
            except Exception as exc:
                print(f"⚠️  Could not load {f}: {exc}", file=sys.stderr)

    if not frames:
        print("❌ No climate data found.", file=sys.stderr)
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_sanitary_data(sanitary_dir, diseases):
    """
    Load sanitary CSV files for all diseases.

    Expected CSV format: date, city, disease, confirmed, suspected, deaths.

    :param sanitary_dir: Directory containing sanitary CSVs.
    :param diseases: List of target diseases.
    :return: pandas DataFrame with sanitary data.

    Example:
        >>> df = load_sanitary_data("data/sanitaires/", ["paludisme"])
        >>> "confirmed" in df.columns or df.empty
        True
    """
    frames = []
    import glob

    for f in sorted(glob.glob(os.path.join(sanitary_dir, "*.csv"))):
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            frames.append(df)
        except Exception as exc:
            print(f"⚠️  Could not load {f}: {exc}", file=sys.stderr)

    if not frames:
        print("⚠️  No sanitary data found. Continuing without.", file=sys.stderr)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "disease" in combined.columns:
        combined = combined[combined["disease"].str.lower().isin(diseases)]
    return combined


def load_scraped_data(scraped_dir="data/scraped/"):
    """
    Load and aggregate web-scraped alert data.

    :param scraped_dir: Directory containing scraped CSV files.
    :return: pandas DataFrame with aggregated alerts.

    Example:
        >>> df = load_scraped_data("data/scraped/")
        >>> isinstance(df, pd.DataFrame)
        True
    """
    frames = []
    import glob

    if not os.path.exists(scraped_dir):
        return pd.DataFrame()

    for f in sorted(glob.glob(os.path.join(scraped_dir, "*.csv"))):
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as exc:
            print(f"⚠️  Could not load {f}: {exc}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def handle_missing_values(df, strategy="interpolate", max_missing_ratio=0.3):
    """
    Handle missing values according to the specified strategy.

    :param df: Input DataFrame.
    :param strategy: One of 'interpolate', 'forward_fill', 'drop'.
    :param max_missing_ratio: Max ratio of missing values before dropping a column.
    :return: Cleaned DataFrame.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        >>> cleaned = handle_missing_values(df, max_missing_ratio=0.5)
        >>> "b" not in cleaned.columns
        True
    """
    # Drop columns with too many missing values
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    if cols_to_drop:
        print(f"🗑️  Dropping columns with >{max_missing_ratio*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Apply strategy to remaining columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    elif strategy == "forward_fill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif strategy == "drop":
        df = df.dropna(subset=numeric_cols)

    return df


def create_lstm_sequences(df, sequence_length=12, target_col="confirmed"):
    """
    Create temporal sequences for LSTM training from a time-ordered DataFrame.

    :param df: DataFrame sorted by date with numeric features.
    :param sequence_length: Number of timesteps per sequence.
    :param target_col: Column to predict.
    :return: Tuple (X, y) as numpy arrays.

    Example:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({"temp": range(20), "confirmed": range(20)})
        >>> X, y = create_lstm_sequences(df, sequence_length=5, target_col="confirmed")
        >>> X.shape[1] == 5
        True
    """
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    if target_col not in df.columns:
        print(f"⚠️  Target column '{target_col}' not found.", file=sys.stderr)
        return np.array([]), np.array([])

    features = df[feature_cols].values
    target = df[target_col].values

    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])

    return np.array(X), np.array(y)


def create_cnn_images(df, grid_size=(32, 32), channels=None):
    """
    Create spatial images for CNN from aggregated data.

    Each channel corresponds to a variable (temperature, precipitation, etc).
    Data is spatially interpolated onto a regular grid.

    :param df: DataFrame with lat, lon, and variable columns.
    :param grid_size: Tuple (height, width) of the output grid.
    :param channels: List of column names to use as channels.
    :return: numpy array of shape (n_timesteps, height, width, n_channels).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"lat": [0], "lon": [0], "temp": [25], "precip": [10]})
        >>> imgs = create_cnn_images(df, grid_size=(8, 8), channels=["temp", "precip"])
        >>> imgs.shape[-1] == 2
        True
    """
    if channels is None:
        channels = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in ("lat", "lon")]

    height, width = grid_size
    n_channels = len(channels)

    # Placeholder: in real implementation, spatially interpolate
    # data onto the grid using scipy or rasterio
    images = np.zeros((1, height, width, n_channels))
    print(f"🖼️  Created CNN image tensor: {images.shape}")
    return images


def preprocess(config_path):
    """
    Main preprocessing pipeline: loads, cleans, and structures all data.
    """
    config = load_config(config_path)

    # Récupération des régions (priorité aux provinces, fallback aux cities)
    regions_dict = config.get("provinces", config.get("cities", {}))
    regions = [r.lower() for r in regions_dict.keys()]
    
    diseases = config["diseases"]
    paths = config["paths"]
    missing_cfg = config["missing_values"]
    seq_length = config["temporal"]["sequence_length"]

    print("=" * 60)
    print("🔄 PREPROCESSING PIPELINE (RDC Global)")
    print("=" * 60)

    # 1. Load climate data
    print("\n📊 Loading climate data for available regions...")
    climate_df = load_climate_csv(paths["climate_raw"], regions)
    if not climate_df.empty:
        climate_df = handle_missing_values(
            climate_df,
            strategy=missing_cfg["strategy"],
            max_missing_ratio=missing_cfg["max_missing_ratio"],
        )
        print(f"   ✅ Climate: {len(climate_df)} rows for {climate_df['city'].nunique()} regions")

    # 2. Load sanitary data
    print("\n🏥 Loading sanitary data...")
    sanitary_df = load_sanitary_data(paths["sanitaires_raw"], diseases)
    if not sanitary_df.empty:
        sanitary_df = handle_missing_values(sanitary_df, strategy=missing_cfg["strategy"])
        print(f"   ✅ Sanitary: {len(sanitary_df)} rows")

    # 3. Load scraped data
    print("\n🌐 Loading scraped data...")
    scraped_df = load_scraped_data(paths.get("scraped", "data/scraped/"))
    if not scraped_df.empty:
        print(f"   ✅ Scraped: {len(scraped_df)} rows")

    # 4. Create output directory
    output_dir = paths["processed"]
    os.makedirs(output_dir, exist_ok=True)

    # 5. Save processed data
    result = {}

    if not climate_df.empty:
        climate_out = os.path.join(output_dir, "climate_processed.csv")
        climate_df.to_csv(climate_out, index=False)
        print(f"\n💾 Climate saved: {climate_out}")

    if not sanitary_df.empty:
        sanitary_out = os.path.join(output_dir, "sanitary_processed.csv")
        sanitary_df.to_csv(sanitary_out, index=False)
        print(f"💾 Sanitary saved: {sanitary_out}")

        # Create LSTM sequences per region
        for region in regions:
            region_data = sanitary_df[sanitary_df["city"].str.lower() == region]
            if len(region_data) > seq_length:
                X, y = create_lstm_sequences(region_data, sequence_length=seq_length)
                if X.size > 0:
                    safe_name = region.replace(" ", "_")
                    np.save(os.path.join(output_dir, f"{safe_name}_X_lstm.npy"), X)
                    np.save(os.path.join(output_dir, f"{safe_name}_y_lstm.npy"), y)
                    result[f"{region}_lstm"] = {"X_shape": X.shape, "y_shape": y.shape}
                    print(f"💾 LSTM sequences for {region}: X={X.shape}, y={y.shape}")

    if not scraped_df.empty:
        scraped_out = os.path.join(output_dir, "scraped_processed.csv")
        scraped_df.to_csv(scraped_out, index=False)
        print(f"💾 Scraped saved: {scraped_out}")

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess health data for model training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocess.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    preprocess(args.config)
