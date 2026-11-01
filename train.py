import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Key ML Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from data import load_data

def build_preprocessor(categorical_features, numeric_features):
    """
    Creates a scikit-learn ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Trains multiple regression models, evaluates their performance,
    and returns a comparison dataframe along with the trained pipelines.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7),
        "Support Vector Regressor": SVR(C=10.0, epsilon=0.1),
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
        })
        trained_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    return results_df, trained_pipelines

def analyze_feature_importance(best_name, best_pipeline, categorical_features, numeric_features):
    """
    Extracts and prints the top influential features for the winning model.
    """
    preprocessor = best_pipeline.named_steps["preprocessor"]
    regressor = best_pipeline.named_steps["regressor"]

    # Retrieve one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names

    print("\n--- Feature Impact Analysis ---")
    if hasattr(regressor, "coef_"):
        # For Linear / Ridge models
        coefs = pd.Series(regressor.coef_, index=all_feature_names).sort_values(key=abs, ascending=False)
        print("Top 8 Influential Coefficients (Linear Weights):")
        for feat, val in coefs.head(8).items():
            direction = "+" if val >= 0 else "-"
            print(f"  {feat:<35} : {direction}{abs(val):.2f}")
    elif hasattr(regressor, "feature_importances_"):
        # For Tree / Ensemble models
        importances = pd.Series(regressor.feature_importances_, index=all_feature_names).sort_values(ascending=False)
        print("Top 8 Feature Importances:")
        for feat, val in importances.head(8).items():
            print(f"  {feat:<35} : {val*100:.1f}%")

def main():
    print("==================================================================")
    print("  STUDENT MARKS PREDICTION - MODEL TRAINING & BENCHMARKING PIPELINE")
    print("==================================================================")

    # 1. Load dataset
    df = load_data()
    print(f"\nSuccessfully loaded dataset with {df.shape[0]} samples and {df.shape[1]} features.")

    # 2. Define Features & Target
    # Target: math_score (marks out of 100)
    target_column = "math_score"
    
    categorical_features = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
    ]
    numeric_features = ["reading_score", "writing_score"]

    X = df[categorical_features + numeric_features]
    y = df[target_column]

    # 3. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dataset Split -> Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Create Preprocessor Pipeline
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    # 5. Train & Evaluate Models
    print("\nTraining and evaluating candidate machine learning models...")
    results_df, trained_pipelines = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    # Display comparison table
    print("\n" + "=" * 62)
    print("           MODEL BENCHMARK COMPARISON (Test Set)")
    print("=" * 62)
    print(results_df.to_string(index=False))
    print("=" * 62)

    # 6. Select the Best Model
    best_name = results_df.iloc[0]["Model"]
    best_r2 = results_df.iloc[0]["R2 Score"]
    best_mae = results_df.iloc[0]["MAE"]
    best_pipeline = trained_pipelines[best_name]

    print(f"\n[WINNER] Best Performing Model: {best_name}")
    print(f"   -> R-squared (Variance Explained): {best_r2 * 100:.2f}%")
    print(f"   -> Mean Absolute Error (MAE):      +/- {best_mae} marks")

    # 7. Analyze Feature Importances
    analyze_feature_importance(best_name, best_pipeline, categorical_features, numeric_features)

    # 8. Save Model and Metadata
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "student_marks_model.joblib")
    metrics_path = os.path.join("artifacts", "metrics.json")

    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "features": {
            "categorical": categorical_features,
            "numeric": numeric_features,
        },
        "target": target_column,
    }, model_path)
    print(f"\nTrained model pipeline saved to: {model_path}")

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump({
            "best_model": best_name,
            "r2_score": best_r2,
            "mae": best_mae,
            "all_results": results_df.to_dict(orient="records"),
        }, f, indent=4)
    print(f"Benchmark metrics saved to: {metrics_path}")

    # 9. Quick Verification with Sample Inference
    print("\n--- Verifying Inference on Sample Students ---")
    sample_students = pd.DataFrame([
        {
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "reading_score": 72,
            "writing_score": 74,
        },
        {
            "gender": "male",
            "race_ethnicity": "group E",
            "parental_level_of_education": "associate's degree",
            "lunch": "standard",
            "test_preparation_course": "completed",
            "reading_score": 85,
            "writing_score": 88,
        },
        {
            "gender": "female",
            "race_ethnicity": "group A",
            "parental_level_of_education": "some high school",
            "lunch": "free/reduced",
            "test_preparation_course": "none",
            "reading_score": 55,
            "writing_score": 52,
        }
    ])

    sample_predictions = best_pipeline.predict(sample_students)
    for i, pred in enumerate(sample_predictions):
        profile = sample_students.iloc[i]
        print(f"Student {i+1} ({profile['gender']}, prep: {profile['test_preparation_course']}, read: {profile['reading_score']}, write: {profile['writing_score']}) -> Predicted Math Mark: {pred:.1f}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
