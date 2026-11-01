import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from predict import predict_student, load_prediction_pipeline
import db

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "student-marks-secret-key-2026")

# Ensure model exists on startup
try:
    load_prediction_pipeline()
    print("[App] ML Model Pipeline loaded successfully.")
except Exception as e:
    print(f"[App Warning] ML Model not loaded: {e}. Run 'python train.py' to generate.")

@app.route("/")
def index():
    """Renders the prediction page, showing ONLY the active user's own data."""
    # Check if a client name is saved in the user's session
    client_name = session.get("current_client")
    client_history = db.get_client_history(client_name) if client_name else []
    
    # Check for one-time result from PRG (Post-Redirect-Get)
    prediction_data = session.pop("prediction_result", None) or {}
    active_client = prediction_data.get("current_client") or client_name

    if not client_history and active_client:
        client_history = db.get_client_history(active_client)

    return render_template(
        "index.html",
        current_client=active_client,
        predicted_score=prediction_data.get("predicted_score"),
        client_history=client_history,
    )

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles score prediction requests and persists user data & predicted scores to MongoDB.
    Supports both JSON API calls and HTML Form Submissions.
    Uses Post-Redirect-Get (PRG) for form submissions to prevent resubmission on reload.
    """
    is_json = request.is_json
    data = request.get_json() if is_json else request.form

    client_name = data.get("client_name", "").strip() or "Anonymous Student"
    
    features = {
        "gender": data.get("gender", "female"),
        "race_ethnicity": data.get("race_ethnicity", "group B"),
        "parental_level_of_education": data.get("parental_level_of_education", "some college"),
        "lunch": data.get("lunch", "standard"),
        "test_preparation_course": data.get("test_preparation_course", "none"),
        "reading_score": float(data.get("reading_score", 70)),
        "writing_score": float(data.get("writing_score", 70)),
    }

    try:
        # Run ML inference
        predicted_scores = predict_student(features)
        predicted_score = predicted_scores[0]

        # Save to MongoDB
        record_id = db.save_prediction(client_name, features, predicted_score)

        # Retrieve all past records for this client
        client_history = db.get_client_history(client_name)

        # Store active client in session so reloads retain only this user's view
        session["current_client"] = client_name

        if is_json:
            return jsonify({
                "success": True,
                "client_name": client_name,
                "predicted_score": predicted_score,
                "record_id": record_id,
                "history": client_history,
            })
        else:
            # Post-Redirect-Get: store result in session and redirect to GET /
            session["prediction_result"] = {
                "predicted_score": predicted_score,
                "current_client": client_name,
                "client_history": client_history,
            }
            return redirect(url_for("index"))

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if is_json:
            return jsonify({"success": False, "error": error_msg}), 400
        return render_template("index.html", error=error_msg, client_history=[])

@app.route("/history/<client_name>", methods=["GET"])
def history(client_name):
    """Returns past prediction records strictly for a specific client."""
    records = db.get_client_history(client_name)
    return jsonify({"client_name": client_name, "records": records})

@app.route("/clear-session", methods=["POST", "GET"])
def clear_session():
    """Clears the active client session so another user can use the app privately."""
    session.pop("current_client", None)
    session.pop("prediction_result", None)
    return redirect(url_for("index"))

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for deployment monitoring."""
    db_status = "connected" if db.get_database() is not None else "disconnected"
    return jsonify({"status": "healthy", "database": db_status})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
