# Student Marks Prediction Machine Learning Project

This project builds an end-to-end Machine Learning pipeline to predict student exam marks using the **Students Performance in Exams** dataset from Kaggle (`spscientist/students-performance-in-exams`).

---

## 📊 Dataset Overview

The dataset consists of 1,000 student records with demographic and academic indicators:
- **Demographic & Contextual Features**:
  - `gender`: Gender (`male`, `female`)
  - `race_ethnicity`: Ethnic group (`group A`, `group B`, `group C`, `group D`, `group E`)
  - `parental_level_of_education`: Highest education level of parents
  - `lunch`: Lunch support type (`standard`, `free/reduced`)
  - `test_preparation_course`: Preparation course status (`completed`, `none`)
- **Exam Scores (0 - 100)**:
  - `reading_score`
  - `writing_score`
  - `math_score` *(Target Variable)*

---

## 🚀 Model Benchmarking Results (Holdout Test Set)

We evaluated 7 key Machine Learning regression models:

| Model | $R^2$ Score (Variance Explained) | MAE (Mean Absolute Error) | RMSE |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (L2)** | **88.06%** | **±4.21 marks** | **5.39** |
| **Linear Regression** | **88.04%** | **±4.21 marks** | **5.39** |
| **Gradient Boosting** | **87.33%** | **±4.26 marks** | **5.55** |
| **Random Forest Regressor** | **85.04%** | **±4.70 marks** | **6.03** |
| **Hist Gradient Boosting** | **84.25%** | **±4.65 marks** | **6.19** |
| **Support Vector Regressor (SVR)** | **82.03%** | **±4.80 marks** | **6.61** |
| **K-Nearest Neighbors (KNN)** | **79.37%** | **±5.54 marks** | **7.09** |

> **Top Performer**: **Ridge Regression** achieves an $R^2$ of **88.06%** with an average error of only **±4.21 marks** out of 100. It regularizes multicollinearity between writing and reading scores while incorporating demographic factors.

---

## 📁 Project Structure

```
deploy/
├── Dockerfile                   # Container definition for Python Flask ML web service
├── docker-compose.yml           # Multi-container orchestration (Web App + MongoDB in one run)
├── .dockerignore                # Build exclusions (ignoring venv, cache, etc.)
├── app.py                       # Flask web backend with /predict and /history endpoints
├── db.py                        # MongoDB connection and prediction history persistence
├── templates/
│   └── index.html               # Clean, responsive web interface
├── data.py                      # Kaggle dataset download and preprocessing
├── train.py                     # ML pipeline: preprocessing, model training & benchmark
├── predict.py                   # Model inference logic
├── StudentsPerformance.csv     # Local cached copy of the dataset
├── requirements.txt             # Project dependencies for deployment
├── Procfile                     # Cloud PaaS deployment process definition
├── .env.example                 # MongoDB configuration template
└── artifacts/
    ├── student_marks_model.joblib # Serialized winning ML pipeline (Ridge Regression)
    └── metrics.json             # Benchmark scores and evaluation details
```

---

## 🐳 One-Command Docker Deployment (Recommended)

To spin up **everything (Flask ML Application + MongoDB + Persistent Volume)** in one run:

```bash
docker compose up --build
```

- Open **`http://localhost:5000`** in your browser.
- **MongoDB** runs automatically inside container `student_mongodb`.
- **Flask ML App** runs automatically inside container `student_predictor_web`.
- Student history and past scores are persistently stored in the `mongodb_data` Docker volume.

To stop the containers:
```bash
docker compose down
```

---

## 🛠️ How to Run

### 1. Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run the Web Application (Frontend + Backend + MongoDB)
```powershell
python app.py
```
Open **`http://localhost:5000`** in your browser.

- Enter student details and click **Predict Math Marks & Save to MongoDB**.
- If a client makes predictions multiple times, all previous predictions along with input scores and predicted marks are stored and displayed in their history table.

### 3. MongoDB Configuration
By default, the application connects to local MongoDB at `mongodb://localhost:27017/`.
To connect to MongoDB Atlas (Cloud) for deployment:
1. Create a `.env` file based on `.env.example`.
2. Set `MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/student_performance_db`.

---

## 💻 CLI Commands (Optional)

- **Train / Benchmark ML Models**:
  ```powershell
  python train.py
  ```
- **CLI Model Prediction**:
  ```powershell
  python predict.py
  ```
