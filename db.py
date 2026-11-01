import os
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Connection URI (Defaults to local MongoDB, or uses MONGO_URI env var for deployment/Atlas)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB_NAME", "student_performance_db")
COLLECTION_NAME = "predictions"

_client = None
_db = None

def get_database():
    """Returns a connected MongoDB database instance, retrying if not yet connected."""
    global _client, _db
    if _db is not None:
        return _db
    try:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.server_info()
        _client = client
        _db = client[DB_NAME]
        print(f"[MongoDB] Successfully connected to database: '{DB_NAME}' at {uri}")
        return _db
    except Exception as e:
        print(f"[MongoDB Warning] Connection attempt to {os.getenv('MONGO_URI', 'mongodb://localhost:27017/')} failed: {e}")
        return None

def save_prediction(client_name, features, predicted_score):
    """
    Saves the user data and the predicted score to MongoDB.
    
    Parameters:
        client_name (str): The client or student name/identifier.
        features (dict): Student features (gender, scores, etc.).
        predicted_score (float): The model's predicted math score.
        
    Returns:
        str: ID of the inserted record or None if DB unavailable.
    """
    db = get_database()
    if db is None:
        return None

    record = {
        "client_name": client_name.strip(),
        "gender": features.get("gender"),
        "race_ethnicity": features.get("race_ethnicity"),
        "parental_level_of_education": features.get("parental_level_of_education"),
        "lunch": features.get("lunch"),
        "test_preparation_course": features.get("test_preparation_course"),
        "reading_score": float(features.get("reading_score", 0)),
        "writing_score": float(features.get("writing_score", 0)),
        "predicted_math_score": float(predicted_score),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }

    result = db[COLLECTION_NAME].insert_one(record)
    return str(result.inserted_id)

def get_client_history(client_name):
    """
    Retrieves all past predictions for a specific client (newest first).
    """
    db = get_database()
    if db is None:
        return []

    records = list(
        db[COLLECTION_NAME].find(
            {"client_name": {"$regex": f"^{client_name.strip()}$", "$options": "i"}},
            {"_id": 0}
        ).sort("created_at", -1)
    )
    return records

def get_all_recent(limit=10):
    """
    Retrieves recent predictions across all clients.
    """
    db = get_database()
    if db is None:
        return []

    records = list(
        db[COLLECTION_NAME].find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    )
    return records
