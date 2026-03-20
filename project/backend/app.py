"""
app.py  –  Multi-Disease Prediction System  (Flask Backend)
Run:  python app.py
"""

import os
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

load_dotenv()

from routes.auth    import auth_bp
from routes.predict import predict_bp
from routes.metrics import metrics_bp

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── JWT config ────────────────────────────────────────────────────────────────
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 86400   # 24 hours

jwt = JWTManager(app)

# ── Blueprints ────────────────────────────────────────────────────────────────
app.register_blueprint(auth_bp,    url_prefix="/api")
app.register_blueprint(predict_bp, url_prefix="/api")
app.register_blueprint(metrics_bp, url_prefix="/api")

@app.route("/")
def index():
    return {"message": "Multi-Disease Prediction API is running ✅", "version": "1.0"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
