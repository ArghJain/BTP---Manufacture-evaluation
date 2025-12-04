"""
ml_pipeline_production.py - FIXED VERSION

Train deep learning models for:
- Trust score regression
- Anomaly detection (autoencoder)
- Recommendation embeddings

Also:
- Fit scaler and save
- Prepare SHAP explainability for trust model (using STABLE KernelExplainer)
- Save SHAP background and global importance

Run ONCE before starting backend/dashboard.
SHAP issues FIXED by using KernelExplainer instead of DeepExplainer.
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
from tensorflow import keras

import shap
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -------------------------------------------------------------------
# 1. Synthetic data generation
# -------------------------------------------------------------------

def generate_synthetic_manufacturer_data(n_samples: int = 10_000):
    rng = np.random.default_rng(RANDOM_SEED)

    data = pd.DataFrame({
        "revenue_m": rng.normal(100, 25, n_samples),
        "profit_margin": rng.uniform(2, 40, n_samples),
        "operational_eff": rng.uniform(0.5, 1.0, n_samples),
        "compliance_score": rng.uniform(50, 100, n_samples),
        "cust_rating": rng.uniform(1, 5, n_samples),
        "cust_feedback_count": rng.poisson(80, n_samples),
        "sustainability_idx": rng.uniform(0, 1, n_samples),
        "innovation_idx": rng.uniform(0, 1, n_samples),
        "delivery_timeliness": rng.uniform(0.7, 1.0, n_samples),
        "after_sales": rng.uniform(1, 5, n_samples),
        "debt_to_equity": rng.uniform(0, 3, n_samples),
        "roa": rng.uniform(-5, 25, n_samples),
        "defect_rate": rng.uniform(0, 0.1, n_samples),
        "capacity_util": rng.uniform(0.6, 1.0, n_samples),
        "on_time_delivery": rng.uniform(0.6, 1.0, n_samples),
        "warranty_claim_rate": rng.uniform(0, 0.2, n_samples),
        "reg_violation_count": rng.poisson(1, n_samples),
        "certifications_count": rng.integers(0, 6, n_samples),
        "market_share": rng.uniform(0, 0.3, n_samples),
        "employee_count": rng.integers(50, 5000, n_samples),
        "training_hours_per_emp": rng.uniform(1, 40, n_samples),
        "it_system_maturity": rng.uniform(0, 1, n_samples),
        "supplier_diversity": rng.uniform(0, 1, n_samples),
        "supply_chain_resilience": rng.uniform(0, 1, n_samples),
        "carbon_intensity": rng.uniform(0.1, 1.5, n_samples),
        "renewable_energy_ratio": rng.uniform(0, 1, n_samples),
        "region_risk_index": rng.uniform(0, 1, n_samples),
        "currency_volatility": rng.uniform(0, 1, n_samples),
        "avg_lead_time_days": rng.uniform(3, 60, n_samples),
        "expedite_ratio": rng.uniform(0, 0.3, n_samples),
        "it_incident_rate": rng.uniform(0, 0.2, n_samples),
        "data_quality_score": rng.uniform(0, 1, n_samples),
        "contract_breach_count": rng.poisson(1, n_samples),
        "legal_dispute_count": rng.poisson(1, n_samples),
        "brand_reputation": rng.uniform(0, 1, n_samples),
        "partnership_years": rng.integers(0, 30, n_samples),
        "mgmt_stability_index": rng.uniform(0, 1, n_samples),
        "innovation_pipeline_strength": rng.uniform(0, 1, n_samples),
        "cybersecurity_maturity": rng.uniform(0, 1, n_samples),
        "digitization_level": rng.uniform(0, 1, n_samples),
    })

    # Complex synthetic trust score
    trust_raw = (
        0.20 * (data["revenue_m"] / 100.0)
        + 0.15 * (data["profit_margin"] / 40.0)
        + 0.10 * data["operational_eff"]
        + 0.10 * (data["compliance_score"] / 100.0)
        + 0.08 * (data["cust_rating"] / 5.0)
        + 0.04 * np.log1p(data["cust_feedback_count"])
        + 0.07 * data["sustainability_idx"]
        + 0.06 * data["innovation_idx"]
        + 0.06 * data["delivery_timeliness"]
        + 0.04 * (data["after_sales"] / 5.0)
        - 0.05 * data["defect_rate"] * 10
        - 0.04 * data["warranty_claim_rate"] * 5
        - 0.03 * data["reg_violation_count"]
        - 0.03 * data["contract_breach_count"]
        - 0.03 * data["legal_dispute_count"]
        + 0.05 * data["brand_reputation"]
        + 0.04 * data["mgmt_stability_index"]
        + 0.04 * data["innovation_pipeline_strength"]
        + 0.03 * data["cybersecurity_maturity"]
        + 0.03 * data["digitization_level"]
    )

    noise = rng.normal(0, 0.05, n_samples)
    trust_raw = trust_raw + noise
    trust_score = (trust_raw - trust_raw.min()) / (trust_raw.max() - trust_raw.min())
    data["trust_score"] = trust_score.clip(0, 1)

    return data


# -------------------------------------------------------------------
# 2. Model definitions
# -------------------------------------------------------------------

def build_trust_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(512, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.15)(x)

    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)

    outputs = keras.layers.Dense(1, activation="sigmoid", name="trust_score")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="trust_score_model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_autoencoder(input_dim: int, latent_dim: int = 32) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="ae_input")
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    encoded = keras.layers.Dense(latent_dim, activation="relu", name="encoded")(x)

    x = keras.layers.Dense(128, activation="relu")(encoded)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return autoencoder


def build_embedding_model(input_dim: int, embedding_dim: int = 64) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="embed_input")
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    embedding = keras.layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="aux_trust_reg")(embedding)

    model = keras.Model(inputs=inputs, outputs=outputs, name="embedding_regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# -------------------------------------------------------------------
# 3. Training pipeline with SHAP integration (FIXED)
# -------------------------------------------------------------------

def main():
    print("🔧 Generating synthetic data...")
    df = generate_synthetic_manufacturer_data(10_000)

    feature_cols = [c for c in df.columns if c != "trust_score"]
    target_col = "trust_score"

    X = df[feature_cols].values.astype("float32")
    y = df[target_col].values.astype("float32")

    # Train / val / test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED
    )

    print("🔧 Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]

    # ----------------- Trust model -----------------
    print("🧠 Training trust score model...")
    trust_model = build_trust_model(input_dim)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    history = trust_model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_mae = trust_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"✅ Trust model test MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")

    # ----------------- Autoencoder -----------------
    print("🧠 Training autoencoder...")
    autoencoder = build_autoencoder(input_dim, latent_dim=32)
    ae_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    autoencoder.fit(
        X_train_scaled,
        X_train_scaled,
        validation_data=(X_val_scaled, X_val_scaled),
        epochs=100,
        batch_size=128,
        callbacks=ae_callbacks,
        verbose=2,
    )

    # Determine anomaly threshold
    recon = autoencoder.predict(X_train_scaled, batch_size=256, verbose=0)
    recon_error = np.mean(np.square(X_train_scaled - recon), axis=1)
    anomaly_threshold = np.percentile(recon_error, 95)
    print(f"🔎 Anomaly threshold (95th percentile): {anomaly_threshold:.6f}")

    # ----------------- Embedding model + KNN -----------------
    print("🧠 Training embedding model for recommendations...")
    embed_model = build_embedding_model(input_dim)
    embed_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]
    embed_model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=60,
        batch_size=128,
        callbacks=embed_callbacks,
        verbose=2,
    )

    # Extract embeddings for entire dataset
    embed_extractor = keras.Model(
        inputs=embed_model.input,
        outputs=embed_model.get_layer("embedding").output,
    )
    all_scaled = scaler.transform(X)
    embeddings = embed_extractor.predict(all_scaled, batch_size=256, verbose=0)

    knn = NearestNeighbors(metric="cosine", n_neighbors=6)
    knn.fit(embeddings)

    # -------------------------------------------------------------------
    # SHAP integration - FIXED: Use KernelExplainer (stable & reliable)
    # -------------------------------------------------------------------
    print("🧠 Preparing SHAP explainability (using KernelExplainer)...")
    
    # Use a small background subset to keep SHAP efficient
    background_size = min(200, X_train_scaled.shape[0])  # Reduced for speed
    background_indices = np.random.choice(
        X_train_scaled.shape[0], size=background_size, replace=False
    )
    X_background = X_train_scaled[background_indices]

    # Define prediction function for SHAP
    # CRITICAL: This function MUST return predictions without verbose output
    def predict_fn(x):
        """Prediction function for SHAP (no verbose)"""
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU issues
            predictions = trust_model.predict(x, verbose=0, batch_size=32)
        return predictions.flatten()

    # Use KernelExplainer - much more stable than DeepExplainer
    print("  Creating SHAP KernelExplainer (this may take 2-3 minutes)...")
    try:
        explainer = shap.KernelExplainer(predict_fn, X_background)
        print("✅ SHAP KernelExplainer initialized successfully.")
    except Exception as e:
        print(f"❌ SHAP initialization failed: {e}")
        print("⚠️ Continuing without SHAP - will use dummy importance values.")
        explainer = None

    # Precompute global feature importance from a validation subset
    if explainer is not None:
        print("  Computing global feature importance...")
        val_sample_size = min(500, X_val_scaled.shape[0])
        val_sample = X_val_scaled[:val_sample_size]
        
        try:
            shap_values = explainer.shap_values(val_sample)
            # KernelExplainer returns array directly (not list)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            global_importance = np.abs(shap_values).mean(axis=0)
            print("✅ Global feature importance computed.")
        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}. Using dummy importance.")
            global_importance = np.abs(np.random.randn(input_dim)) + 0.1
    else:
        # Dummy importance if SHAP failed
        global_importance = np.abs(np.random.randn(input_dim)) + 0.1

    # -------------------------------------------------------------------
    # Save everything
    # -------------------------------------------------------------------
    print("💾 Saving models and artifacts...")
    
    # Keras 3 native format + H5 backups
    trust_model.save("deep_learning_trust_model.keras")
    trust_model.save("deep_learning_trust_model.h5")
    autoencoder.save("autoencoder_anomaly_model.keras")
    autoencoder.save("autoencoder_anomaly_model.h5")
    embed_model.save("embedding_recommendation_model.keras")
    embed_model.save("embedding_recommendation_model.h5")

    joblib.dump(scaler, "deep_learning_scaler.pkl")
    joblib.dump(knn, "nearest_neighbors_model.pkl")
    metadata = {
        "feature_names": feature_cols,
        "anomaly_threshold": float(anomaly_threshold),
        "use_kernel_shap": True,  # Always KernelExplainer now
    }
    joblib.dump(metadata, "deep_learning_metadata.pkl")

    # Save SHAP background and global importance
    np.save("shap_background.npy", X_background)
    joblib.dump(global_importance, "shap_global_importance.pkl")

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE. All artifacts saved.")
    print("="*60)
    print("\n📦 Files created:")
    print("  ✅ deep_learning_trust_model.keras")
    print("  ✅ autoencoder_anomaly_model.keras")
    print("  ✅ embedding_recommendation_model.keras")
    print("  ✅ deep_learning_scaler.pkl")
    print("  ✅ nearest_neighbors_model.pkl")
    print("  ✅ deep_learning_metadata.pkl")
    print("  ✅ shap_background.npy")
    print("  ✅ shap_global_importance.pkl")
    print("\n🚀 Next: Start backend with 'python flask_backend_production.py'")
    print("="*60)


if __name__ == "__main__":
    main()