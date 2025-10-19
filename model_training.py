import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings

warnings.filterwarnings("ignore")


class AdvancedStudentHealthModel:
    def __init__(self):
        self.models = {}
        self.encoders = {}  # Store LabelEncoders for each categorical column
        self.scaler = StandardScaler()
        self.best_model_name = None
        self.best_model = None
        self.best_score = 0
        self.feature_names = None
        self.performance_results = {}

    def preprocess_data(self, df):
        print("Columns found:", df.columns.tolist())
        print("First 5 rows:\n", df.head())

        # Drop empty rows
        df_clean = df.dropna(how="all")

        # Remove stray header rows
        df_clean = df_clean[df_clean["Gender"] != "Gender"]

        # Strip column names
        df_clean.columns = df_clean.columns.str.strip()

        # Categorical columns to encode
        categorical_cols = [
            "Gender",
            "Sleep Duration",
            "Dietary Habits",
            "Family History of Mental Illness",
            "Have you ever had suicidal thoughts ?",
        ]

        # Encode each categorical column
        for col in categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.encoders[col] = le  # Save encoder
            else:
                print(f"⚠️ Warning: Column '{col}' not found in CSV!")

        # Convert numeric columns
        numeric_cols = ["Age", "Academic Pressure", "Study Satisfaction", "Study Hours", "Financial Stress"]
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Drop rows with NaN after conversion
        df_clean = df_clean.dropna()

        print("\n✅ Data after preprocessing:")
        print(df_clean.head())
        return df_clean

    def train_all_models(self, csv_path):
        df = pd.read_csv(csv_path)
        self.feature_names = [col for col in df.columns if col.strip() != "Depression"]

        df_clean = self.preprocess_data(df)
        X = df_clean[self.feature_names]
        y = df_clean["Depression"]

        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.encoders["Depression"] = le_target

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model configs
        model_configs = {
            "Random Forest": (RandomForestClassifier(random_state=42), {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}),
            "SVM": (SVC(probability=True, random_state=42), {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}),
            "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), {"C": [0.1, 1, 10]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [None, 5, 10]}),
            "K-Nearest Neighbors": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
            "Naive Bayes": (GaussianNB(), {}),
            "AdaBoost": (AdaBoostClassifier(random_state=42), {"n_estimators": [50, 100]}),
        }

        results = {}
        for name, (model, params) in model_configs.items():
            try:
                if params:
                    grid = GridSearchCV(model, params, cv=5, scoring="accuracy")
                    grid.fit(X_train_scaled if name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"] else X_train, y_train)
                    best_model = grid.best_estimator_
                    accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled if name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"] else X_test))
                else:
                    best_model = model.fit(X_train_scaled if name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"] else X_train, y_train)
                    accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled if name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"] else X_test))

                results[name] = {"model": best_model, "accuracy": accuracy}

                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = best_model
                    self.best_model_name = name
            except Exception as e:
                print(f"{name} error: {e}")
                continue

        self.models = results
        self.performance_results = results

        # Save all
        joblib.dump({
            "models": self.models,
            "encoders": self.encoders,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "performance_results": self.performance_results,
        }, "all_models.pkl")

        return results

    def predict(self, input_data):
        if not self.best_model:
            raise ValueError("No trained model available")

        input_df = pd.DataFrame(columns=self.feature_names)
        for col in self.feature_names:
            input_df[col] = [input_data.get(col, 0)]

        # Encode categorical using saved encoders
        categorical_cols = ["Gender", "Dietary Habits", "Family History of Mental Illness", "Have you ever had suicidal thoughts ?"]
        for col in categorical_cols:
            if col in input_df.columns and col in self.encoders:
                val = str(input_data[col])
                le = self.encoders[col]
                input_df[col] = le.transform([val])[0] if val in le.classes_ else le.transform([le.classes_[0]])[0]

        # Sleep Duration conversion
        def get_sleep_hours_input(duration):
            duration = str(duration).lower()
            if "7-8" in duration:
                return 7.5
            elif "5-6" in duration:
                return 5.5
            elif "8+" in duration or "more than 8" in duration:
                return 8.5
            elif "6-7" in duration:
                return 6.5
            elif "4-5" in duration or "less than 5" in duration:
                return 4.5
            else:
                return 6.0

        input_df["Sleep Duration"] = get_sleep_hours_input(input_data["Sleep Duration"])

        # Numeric columns
        numeric_cols = ["Age", "Academic Pressure", "Study Satisfaction", "Study Hours", "Financial Stress"]
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # Prepare input
        X_input = self.scaler.transform(input_df) if self.best_model_name in ["SVM", "Logistic Regression", "K-Nearest Neighbors"] else input_df.values

        # Predict
        prediction = self.best_model.predict(X_input)[0]
        probability = self.best_model.predict_proba(X_input)[0]

        return {
            "prediction": self.encoders["Depression"].inverse_transform([prediction])[0],
            "confidence": max(probability),
            "risk_level": "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low",
            "probabilities": {"No Depression": probability[0], "Depression": probability[1]},
            "best_model_used": self.best_model_name,
        }

    def load_models(self):
        try:
            data = joblib.load("all_models.pkl")
            self.models = data["models"]
            self.encoders = data["encoders"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self.best_model_name = data["best_model_name"]
            self.best_model = data["best_model"]
            self.performance_results = data["performance_results"]
            return True
        except:
            return False
