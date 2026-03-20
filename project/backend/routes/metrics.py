"""
routes/metrics.py  –  Model metrics endpoint
GET /api/model-metrics
"""

import os, json
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

metrics_bp = Blueprint("metrics", __name__)

BASE        = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "saved_models"))
METRICS_PATH = os.path.join(BASE, "metrics.json")


@metrics_bp.route("/model-metrics", methods=["GET"])
@jwt_required()
def model_metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics not found. Run train_models.py first."}), 404
    with open(METRICS_PATH) as f:
        data = json.load(f)
    return jsonify(data)
