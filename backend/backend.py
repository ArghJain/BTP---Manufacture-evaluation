"""
flask_backend_production.py - FIXED VERSION

Loads trained models + scaler + SHAP artifacts.
Exposes REST API for:
- /predict : trust score + local SHAP
- /anomaly : anomaly flag + score
- /recommend : similar manufacturers
- /shap-global : global SHAP importance

SHAP FIXES:
- Handles KernelExplainer (stable)
- Graceful fallback if SHAP unavailable
- Proper error handling
"""

import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow import keras
import shap
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# -------------------------------------------------------------------
# Load artifacts
# -------------------------------------------------------------------

def load_keras_any(path_without_ext: str):
    """
    Try loading Keras model from .keras then .h5
    """
    if os.path.exists(path_without_ext + ".keras"):
        return keras.models.load_model(path_without_ext + ".keras", compile=False)
    elif os.path.exists(path_without_ext + ".h5"):
        return keras.models.load_model(path_without_ext + ".h5", compile=False)
    else:
        raise FileNotFoundError(f"Could not find {path_without_ext}.keras or .h5")


print("🔧 Loading models and artifacts...")

try:
    trust_model = load_keras_any(os.path.join(MODEL_DIR, "deep_learning_trust_model"))
    print("✅ Trust model loaded")
except Exception as e:
    print(f"❌ Error loading trust model: {e}")
    raise

try:
    autoencoder = load_keras_any(os.path.join(MODEL_DIR, "autoencoder_anomaly_model"))
    print("✅ Autoencoder loaded")
except Exception as e:
    print(f"❌ Error loading autoencoder: {e}")
    raise

try:
    embed_model = load_keras_any(os.path.join(MODEL_DIR, "embedding_recommendation_model"))
    print("✅ Embedding model loaded")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    raise

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "deep_learning_scaler.pkl"))
    print("✅ Scaler loaded")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    raise

try:
    knn = joblib.load(os.path.join(MODEL_DIR, "nearest_neighbors_model.pkl"))
    print("✅ KNN model loaded")
except Exception as e:
    print(f"❌ Error loading KNN: {e}")
    raise

try:
    metadata = joblib.load(os.path.join(MODEL_DIR, "deep_learning_metadata.pkl"))
    feature_names = metadata["feature_names"]
    anomaly_threshold = metadata["anomaly_threshold"]
    print("✅ Metadata loaded")
except Exception as e:
    print(f"❌ Error loading metadata: {e}")
    raise

try:
    X_background = np.load(os.path.join(MODEL_DIR, "shap_background.npy"))
    global_importance = joblib.load(os.path.join(MODEL_DIR, "shap_global_importance.pkl"))
    print("✅ SHAP artifacts loaded")
    shap_available = True
except Exception as e:
    print(f"⚠️ SHAP artifacts not available: {e}")
    shap_available = False
    X_background = None
    global_importance = None

# Recreate SHAP explainer (KernelExplainer - stable)
explainer = None
if shap_available and X_background is not None:
    print("🔧 Reconstructing SHAP explainer...")
    try:
        def predict_fn(x):
            """Prediction function for SHAP"""
            with tf.device('/CPU:0'):
                predictions = trust_model.predict(x, verbose=0, batch_size=32)
            return predictions.flatten()
        
        explainer = shap.KernelExplainer(predict_fn, X_background)
        print("✅ SHAP KernelExplainer initialized.")
    except Exception as e:
        print(f"⚠️ SHAP initialization failed: {e}. Local SHAP will be unavailable.")
        explainer = None

# Embedding extractor for recommendations
embed_extractor = keras.Model(
    inputs=embed_model.input,
    outputs=embed_model.get_layer("embedding").output,
)

print("\n" + "="*60)
print("✅ ALL MODELS AND SHAP EXPLAINER LOADED.")
print("="*60 + "\n")

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__)
CORS(app)


def validate_input(payload):
    """
    Expect JSON with keys exactly matching feature_names.
    """
    missing = [f for f in feature_names if f not in payload]
    if missing:
        return False, f"Missing fields: {missing}"
    x = np.array([payload[f] for f in feature_names], dtype="float32").reshape(1, -1)
    return True, x


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    
    Input: JSON with all 40 features
    Output: trust_score + local SHAP explanation (if available)
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "No JSON payload"}), 400

    ok, x_raw = validate_input(data)
    if not ok:
        return jsonify({"error": x_raw}), 400

    x_scaled = scaler.transform(x_raw)
    
    # Prediction
    with tf.device('/CPU:0'):
        y_pred = trust_model.predict(x_scaled, verbose=0)[0, 0]

    # Local SHAP values (if available)
    local_shap = []
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(x_scaled)
            # KernelExplainer returns array directly
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_vals_sample = shap_values.flatten()

            # Sort features by |shap| for nicer display
            abs_vals = np.abs(shap_vals_sample)
            order = np.argsort(-abs_vals)
            local_shap = [
                {
                    "feature": feature_names[i],
                    "shap_value": float(shap_vals_sample[i]),
                    "abs_shap": float(abs_vals[i]),
                    "input_value": float(x_raw[0, i]),
                }
                for i in order
            ]
        except Exception as e:
            print(f"⚠️ SHAP calculation failed: {e}")
            local_shap = []

    response = {
        "trust_score": float(y_pred),
        "local_shap": local_shap,
    }
    return jsonify(response)


@app.route("/anomaly", methods=["POST"])
def anomaly():
    """
    POST /anomaly
    
    Input: JSON with all 40 features
    Output: reconstruction_error + is_anomaly flag
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "No JSON payload"}), 400

    ok, x_raw = validate_input(data)
    if not ok:
        return jsonify({"error": x_raw}), 400

    x_scaled = scaler.transform(x_raw)
    
    with tf.device('/CPU:0'):
        recon = autoencoder.predict(x_scaled, verbose=0)
    
    error = float(np.mean((x_scaled - recon) ** 2))
    is_anomaly = error > anomaly_threshold
    response = {
        "reconstruction_error": error,
        "threshold": float(anomaly_threshold),
        "is_anomaly": bool(is_anomaly),
    }
    return jsonify(response)


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    
    Input: JSON with all 40 features
    Output: Top 5 similar manufacturers (indices + similarity scores)
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "No JSON payload"}), 400

    ok, x_raw = validate_input(data)
    if not ok:
        return jsonify({"error": x_raw}), 400

    x_scaled = scaler.transform(x_raw)
    
    with tf.device('/CPU:0'):
        emb = embed_extractor.predict(x_scaled, verbose=0)
    
    distances, indices = knn.kneighbors(emb, n_neighbors=6)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append(
            {
                "rank": rank,
                "index": int(idx),
                "cosine_distance": float(dist),
                "similarity": float(1.0 - dist),
            }
        )

    return jsonify({"neighbors": results})


@app.route("/shap-global", methods=["GET"])
def shap_global():
    """
    GET /shap-global
    
    Output: Global feature importance (mean |SHAP|) across validation set
    """
    if global_importance is None:
        return jsonify({"error": "Global SHAP importance not available"}), 503

    order = np.argsort(-global_importance)
    data = [
        {
            "feature": feature_names[i],
            "mean_abs_shap": float(global_importance[i]),
        }
        for i in order
    ]
    return jsonify({"global_importance": data})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)