from pathlib import Path
import json
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "iot_autoencoder.keras"
SCALER_PATH = MODEL_DIR / "scaler.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

model = None
scaler = None
threshold = None


def load_artifacts():
    """Load the trained Keras model and its small configuration files."""
    global model, scaler, threshold

    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and THRESHOLD_PATH.exists()):
        return False

    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH)
    with SCALER_PATH.open(encoding="utf-8") as file:
        scaler = json.load(file)
    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])
    return True


def score_window(values):
    values = np.asarray(values, dtype="float32")
    expected = int(scaler["time_steps"])

    if len(values) != expected:
        raise ValueError(f"Please provide exactly {expected} sensor readings.")

    scaled = (values - float(scaler["mean"])) / float(scaler["std"])
    batch = scaled.reshape(1, expected, 1)
    reconstructed = model.predict(batch, verbose=0)
    error = float(np.mean(np.abs(reconstructed - batch)))
    return error, error > threshold, expected


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    required_points = 60

    if SCALER_PATH.exists():
        with SCALER_PATH.open(encoding="utf-8") as file:
            required_points = int(json.load(file)["time_steps"])

    if request.method == "POST":
        if model is None and not load_artifacts():
            error = "Model files are missing. Run 02_Autoencoder_Training.ipynb first."
        else:
            raw_values = request.form.get("sensor_values", "")
            try:
                values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
                reconstruction_error, is_anomaly, _ = score_window(values)
                result = {
                    "is_anomaly": is_anomaly,
                    "error": reconstruction_error,
                    "threshold": threshold,
                }
            except ValueError as exc:
                error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        required_points=required_points,
        github="https://github.com/InfinitePraveen",
        linkedin="https://www.linkedin.com/in/infinitepraveen/",
    )


if __name__ == "__main__":
    app.run(debug=True)
