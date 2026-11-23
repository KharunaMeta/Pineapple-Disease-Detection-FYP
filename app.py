"""
Flask API and web UI: image upload → OpenCV preprocessing → MobileNetV2 binary inference → JSON + confidence UI.
"""
from __future__ import annotations

import os
import traceback

from flask import Flask, jsonify, render_template, request

from model_builder import CLASS_NAMES, infer, load_or_build_model
from preprocessing import preprocess_from_bytes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

_model, _weights_loaded = load_or_build_model(os.environ.get("MODEL_WEIGHTS_PATH"))


def _parse_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": "MobileNetV2",
            "classes": list(CLASS_NAMES),
            "weights_loaded": _weights_loaded,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    multipart/form-data: field `image` (file). Optional fields:
    - histogram_eq: true/false
    - normalize: mobilenet | div255
    """
    try:
        if "image" not in request.files and "file" not in request.files:
            return jsonify({"error": "No image file provided. Use form field 'image' or 'file'."}), 400

        file_storage = request.files.get("image") or request.files.get("file")
        if not file_storage or file_storage.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        raw = file_storage.read()
        use_he = _parse_bool(request.form.get("histogram_eq") or request.args.get("histogram_eq"))
        norm_mode = (request.form.get("normalize") or request.args.get("normalize") or "mobilenet").strip().lower()
        if norm_mode not in ("mobilenet", "div255"):
            return jsonify({"error": "normalize must be 'mobilenet' or 'div255'."}), 400

        batch = preprocess_from_bytes(
            raw,
            use_histogram_equalization=use_he,
            normalize_mode=norm_mode,
        )
        probs = infer(_model, batch)[0]
        class_id = int(probs.argmax())
        confidence = float(probs[class_id])
        label = CLASS_NAMES[class_id]

        payload = {
            "prediction": label,
            "class_index": class_id,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100.0, 2),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "histogram_equalization": use_he,
            "normalize_mode": norm_mode,
        }
        if not _weights_loaded:
            payload["warning"] = (
                "Classifier head may be untrained. Set MODEL_WEIGHTS_PATH to a trained .keras/.h5 "
                "for meaningful Healthy vs Heart_Rot predictions."
            )
        return jsonify(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Internal error", "detail": traceback.format_exc()}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
