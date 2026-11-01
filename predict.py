import os
import sys
import joblib
import pandas as pd

MODEL_PATH = os.path.join("artifacts", "student_marks_model.joblib")

def load_prediction_pipeline():
    """Loads the trained model pipeline and metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please run 'python train.py' first to train and save the model."
        )
    return joblib.load(MODEL_PATH)

def predict_student(student_profile):
    """
    Predicts the marks (math_score) for one or more students.
    
    Parameters:
        student_profile: dict or pandas DataFrame containing required student features:
            - gender: 'female' | 'male'
            - race_ethnicity: 'group A' | 'group B' | 'group C' | 'group D' | 'group E'
            - parental_level_of_education: "bachelor's degree" | 'some college' | "master's degree" | "associate's degree" | 'high school' | 'some high school'
            - lunch: 'standard' | 'free/reduced'
            - test_preparation_course: 'none' | 'completed'
            - reading_score: numeric (0 - 100)
            - writing_score: numeric (0 - 100)
            
    Returns:
        list of predicted marks.
    """
    bundle = load_prediction_pipeline()
    pipeline = bundle["pipeline"]

    if isinstance(student_profile, dict):
        df = pd.DataFrame([student_profile])
    elif isinstance(student_profile, pd.DataFrame):
        df = student_profile
    else:
        raise ValueError("student_profile must be a dictionary or a pandas DataFrame.")

    predictions = pipeline.predict(df)
    # Clip between 0 and 100 since marks are bounded
    return [max(0.0, min(100.0, round(float(p), 1))) for p in predictions]

def run_demo():
    print("=" * 65)
    print("       STUDENT MARKS PREDICTION - INFERENCE ENGINE")
    print("=" * 65)

    bundle = load_prediction_pipeline()
    print(f"Loaded Trained Model: {bundle['model_name']}")
    print(f"Target:               {bundle['target']} (Marks / 100)\n")

    test_cases = [
        {
            "name": "Alex (High achiever with test prep)",
            "data": {
                "gender": "male",
                "race_ethnicity": "group E",
                "parental_level_of_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "completed",
                "reading_score": 88,
                "writing_score": 92,
            }
        },
        {
            "name": "Sarah (Balanced student without test prep)",
            "data": {
                "gender": "female",
                "race_ethnicity": "group C",
                "parental_level_of_education": "some college",
                "lunch": "standard",
                "test_preparation_course": "none",
                "reading_score": 70,
                "writing_score": 72,
            }
        },
        {
            "name": "Jordan (Needs academic assistance)",
            "data": {
                "gender": "male",
                "race_ethnicity": "group A",
                "parental_level_of_education": "high school",
                "lunch": "free/reduced",
                "test_preparation_course": "none",
                "reading_score": 45,
                "writing_score": 42,
            }
        }
    ]

    for item in test_cases:
        st = item["data"]
        score = predict_student(st)[0]
        print(f"Profile: {item['name']}")
        print(f"  Gender: {st['gender']} | Prep Course: {st['test_preparation_course']} | Lunch: {st['lunch']}")
        print(f"  Parent Education: {st['parental_level_of_education']}")
        print(f"  Reading Score: {st['reading_score']} | Writing Score: {st['writing_score']}")
        print(f"  --> PREDICTED MATH MARK: {score} / 100")
        print("-" * 65)

if __name__ == "__main__":
    try:
        run_demo()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
