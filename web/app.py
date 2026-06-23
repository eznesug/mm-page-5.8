"""Flask web app for QSAR prediction -- deployed on Railway."""
import os
import sys
from types import ModuleType

import numpy as np
import joblib
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# _loss compatibility patch: the model pickle was created on Windows and
# references `_loss` as a top-level module (not sklearn._loss).
# This must run BEFORE any joblib.load() call.
# ---------------------------------------------------------------------------
try:
    import sklearn._loss.loss as _skl_loss_mod
    import sklearn._loss.link as _skl_link_mod

    _loss_pkg = ModuleType("_loss")
    _loss_pkg.loss = _skl_loss_mod
    _loss_pkg.link = _skl_link_mod
    _loss_pkg.__path__ = []
    _loss_pkg.__dict__.update(vars(_skl_loss_mod))
    _loss_pkg.__dict__.update(vars(_skl_link_mod))
    sys.modules["_loss"] = _loss_pkg
    sys.modules["_loss.loss"] = _skl_loss_mod
    sys.modules["_loss.link"] = _skl_link_mod
    del _loss_pkg, _skl_loss_mod, _skl_link_mod
except ImportError:
    pass  # scikit-learn <1.6 or different layout -- patch not needed

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------------------------------------------------------------------------
# Load model assets
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "unified_qsar_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "feature_order.txt")) as f:
    FEATURES = [line.strip() for line in f.readlines()]

# ---------------------------------------------------------------------------
# Oxidant mapping
# ---------------------------------------------------------------------------
OXIDANT_E0_MAP = {"OH": 2.80, "NO3": 2.40, "O3": 2.07}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            oxidant = request.form.get("oxidant")
            if oxidant not in OXIDANT_E0_MAP:
                return render_template(
                    "predict.html",
                    features=FEATURES,
                    prediction=None,
                    error="Invalid oxidant",
                )

            e0 = OXIDANT_E0_MAP[oxidant]

            values = []
            for feat in FEATURES:
                if feat == "Oxidant_E0":
                    values.append(e0)
                else:
                    val = request.form.get(feat)
                    if val is None or val == "":
                        return render_template(
                            "predict.html",
                            features=FEATURES,
                            prediction=None,
                            error=f"Missing feature: {feat}",
                        )
                    values.append(float(val))

            X = np.array(values).reshape(1, -1)
            X = scaler.transform(X)
            prediction = float(model.predict(X)[0])

        except Exception as e:
            return render_template(
                "predict.html",
                features=FEATURES,
                prediction=None,
                error=str(e),
            )

    return render_template(
        "predict.html", features=FEATURES, prediction=prediction, error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
