"""
routes/auth.py  –  Login / Register endpoints
POST /api/login
POST /api/register
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__)

# ── In-memory user store (replace with DB in production) ─────────────────────
USERS = {
    "admin@medai.com": {
        "name":     "Admin User",
        "password": generate_password_hash("admin123"),
        "role":     "admin",
    }
}

# ── POST /api/register ────────────────────────────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name     = data.get("name", "User")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if email in USERS:
        return jsonify({"error": "Email already registered"}), 409
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    USERS[email] = {
        "name":     name,
        "password": generate_password_hash(password),
        "role":     "user",
    }
    token = create_access_token(identity=email)
    return jsonify({"token": token, "name": name, "email": email}), 201


# ── POST /api/login ───────────────────────────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = USERS.get(email)
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=email)
    return jsonify({
        "token": token,
        "name":  user["name"],
        "email": email,
        "role":  user["role"],
    }), 200
