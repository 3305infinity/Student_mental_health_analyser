# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import joblib
# import warnings
# warnings.filterwarnings('ignore')

# class AdvancedStudentHealthModel:
#     def __init__(self):
#         self.models = {}
#         self.encoders = {}
#         self.scaler = StandardScaler()
#         self.best_model = None
#         self.best_score = 0
#         self.feature_names = None
#         self.performance_results = {}
        
#     def train_all_models(self, csv_path):
#         # Load data
#         df = pd.read_csv(csv_path)
#         print(f"📊 Dataset loaded: {len(df)} records")
        
#         # Store feature names
#         self.feature_names = [col for col in df.columns if col != 'Depression']
        
#         # Preprocessing
#         df_clean = df.copy()
        
#         # Handle categorical columns
#         categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 
#                           'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
        
#         for col in categorical_cols:
#             self.encoders[col] = LabelEncoder()
#             df_clean[col] = self.encoders[col].fit_transform(df_clean[col].astype(str))
        
#         # Convert sleep duration to numerical
#         def get_sleep_hours(duration):
#             duration = str(duration).lower()
#             if '7-8' in duration: return 7.5
#             elif '5-6' in duration: return 5.5
#             elif '8+' in duration or 'more than 8' in duration: return 8.5
#             elif '6-7' in duration: return 6.5
#             elif '4-5' in duration or 'less than 5' in duration: return 4.5
#             else: return 6.0
        
#         df_clean['Sleep Duration'] = df_clean['Sleep Duration'].apply(get_sleep_hours)
        
#         # Prepare features and target
#         X = df_clean[self.feature_names]
#         y = df_clean['Depression']
        
#         # Encode target
#         self.encoders['Depression'] = LabelEncoder()
#         y_encoded = self.encoders['Depression'].fit_transform(y)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#         )
        
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Define all models
#         model_configs = {
#             'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#             'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
#             'SVM': SVC(probability=True, random_state=42),
#             'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#             'Decision Tree': DecisionTreeClassifier(random_state=42),
#             'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
#             'Naive Bayes': GaussianNB(),
#             'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
#         }
        
#         # Train and evaluate all models
#         results = {}
#         for name, model in model_configs.items():
#             try:
#                 if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
#                     model.fit(X_train_scaled, y_train)
#                     y_pred = model.predict(X_test_scaled)
#                     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
#                 else:
#                     model.fit(X_train, y_train)
#                     y_pred = model.predict(X_test)
#                     cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
#                 accuracy = accuracy_score(y_test, y_pred)
#                 train_score = model.score(X_train_scaled if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors'] else X_train, y_train)
                
#                 results[name] = {
#                     'model': model,
#                     'accuracy': accuracy,
#                     'train_score': train_score,
#                     'cv_mean': cv_scores.mean(),
#                     'cv_std': cv_scores.std(),
#                     'predictions': y_pred
#                 }
                
#                 # Update best model
#                 if accuracy > self.best_score:
#                     self.best_score = accuracy
#                     self.best_model = name
                    
#                 print(f"✅ {name}: {accuracy:.3f} (CV: {cv_scores.mean():.3f})")
                
#             except Exception as e:
#                 print(f"❌ {name}: {str(e)}")
#                 continue
        
#         self.models = results
#         self.performance_results = results
        
#         # Save models
#         model_data = {
#             'models': self.models,
#             'encoders': self.encoders,
#             'scaler': self.scaler,
#             'feature_names': self.feature_names,
#             'best_model': self.best_model,
#             'best_score': self.best_score,
#             'performance_results': self.performance_results
#         }
#         joblib.dump(model_data, 'all_models.pkl')
        
#         return results
    
#     def predict(self, input_data):
#         if not self.best_model or self.best_model not in self.models:
#             raise ValueError("No trained model available")
        
#         best_model_info = self.models[self.best_model]
#         model = best_model_info['model']
        
#         # Create input DataFrame with correct column order
#         input_df = pd.DataFrame(columns=self.feature_names)
        
#         # Fill the DataFrame with input data
#         for col in self.feature_names:
#             if col in input_data:
#                 input_df[col] = [input_data[col]]
#             else:
#                 input_df[col] = [0]
        
#         # Preprocess categorical variables
#         for col, encoder in self.encoders.items():
#             if col in input_df.columns and col != 'Depression':
#                 input_value = str(input_data[col])
#                 if input_value in encoder.classes_:
#                     input_df[col] = encoder.transform([input_value])[0]
#                 else:
#                     # Handle unseen categories
#                     if col == 'Dietary Habits':
#                         mapped_value = 'Healthy' if 'Healthy' in input_value else 'Moderate'
#                     elif col == 'Sleep Duration':
#                         mapped_value = '7-8 hours'
#                     else:
#                         mapped_value = encoder.classes_[0]
#                     input_df[col] = encoder.transform([mapped_value])[0]
        
#         # Convert sleep duration
#         def get_sleep_hours_input(duration):
#             duration = str(duration).lower()
#             if '7-8' in duration: return 7.5
#             elif '5-6' in duration: return 5.5
#             elif '8+' in duration or 'more than 8' in duration: return 8.5
#             elif '6-7' in duration: return 6.5
#             elif '4-5' in duration or 'less than 5' in duration: return 4.5
#             else: return 6.0
        
#         input_df['Sleep Duration'] = get_sleep_hours_input(input_data['Sleep Duration'])
        
#         # Ensure numeric types
#         for col in ['Age', 'Academic Pressure', 'Study Satisfaction', 'Study Hours', 'Financial Stress']:
#             input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
#         # Scale if needed
#         if self.best_model in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
#             X_input = self.scaler.transform(input_df)
#         else:
#             X_input = input_df.values
        
#         # Make prediction
#         prediction = model.predict(X_input)[0]
#         probability = model.predict_proba(X_input)[0]
        
#         result = {
#             'prediction': self.encoders['Depression'].inverse_transform([prediction])[0],
#             'confidence': max(probability),
#             'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low',
#             'probabilities': {
#                 'No Depression': probability[0],
#                 'Depression': probability[1]
#             },
#             'best_model_used': self.best_model
#         }
        
#         return result
    
#     def load_models(self):
#         try:
#             model_data = joblib.load('all_models.pkl')
#             self.models = model_data['models']
#             self.encoders = model_data['encoders']
#             self.scaler = model_data['scaler']
#             self.feature_names = model_data['feature_names']
#             self.best_model = model_data['best_model']
#             self.best_score = model_data['best_score']
#             self.performance_results = model_data['performance_results']
#             return True
#         except:
#             return False




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedStudentHealthModel:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        self.feature_names = None
        self.performance_results = {}
        
    def train_all_models(self, csv_path):
        df = pd.read_csv(csv_path)
        self.feature_names = [col for col in df.columns if col != 'Depression']

        df_clean = df.copy()
        categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 
                          'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
        
        for col in categorical_cols:
            self.encoders[col] = LabelEncoder()
            df_clean[col] = self.encoders[col].fit_transform(df_clean[col].astype(str))
        
        # Convert sleep duration to numeric
        def get_sleep_hours(duration):
            duration = str(duration).lower()
            if '7-8' in duration: return 7.5
            elif '5-6' in duration: return 5.5
            elif '8+' in duration or 'more than 8' in duration: return 8.5
            elif '6-7' in duration: return 6.5
            elif '4-5' in duration or 'less than 5' in duration: return 4.5
            else: return 6.0
        
        df_clean['Sleep Duration'] = df_clean['Sleep Duration'].apply(get_sleep_hours)
        
        X = df_clean[self.feature_names]
        y = df_clean['Depression']
        self.encoders['Depression'] = LabelEncoder()
        y_encoded = self.encoders['Depression'].fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_configs = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in model_configs.items():
            try:
                if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean()
                }
                
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = name
            except Exception as e:
                continue
        
        self.models = results
        self.performance_results = results
        joblib.dump({
            'models': self.models,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'performance_results': self.performance_results
        }, 'all_models.pkl')
        
        return results
    
    def predict(self, input_data):
        if not self.best_model or self.best_model not in self.models:
            raise ValueError("No trained model available")
        
        best_model = self.models[self.best_model]['model']
        input_df = pd.DataFrame(columns=self.feature_names)
        
        for col in self.feature_names:
            input_df[col] = [input_data.get(col, 0)]
        
        for col, encoder in self.encoders.items():
            if col in input_df.columns and col != 'Depression':
                val = str(input_data[col])
                if val in encoder.classes_:
                    input_df[col] = encoder.transform([val])[0]
                else:
                    input_df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        def get_sleep_hours_input(duration):
            duration = str(duration).lower()
            if '7-8' in duration: return 7.5
            elif '5-6' in duration: return 5.5
            elif '8+' in duration or 'more than 8' in duration: return 8.5
            elif '6-7' in duration: return 6.5
            elif '4-5' in duration or 'less than 5' in duration: return 4.5
            else: return 6.0
        
        input_df['Sleep Duration'] = get_sleep_hours_input(input_data['Sleep Duration'])
        
        numeric_cols = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Study Hours', 'Financial Stress']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        if self.best_model in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
            X_input = self.scaler.transform(input_df)
        else:
            X_input = input_df.values
        
        prediction = best_model.predict(X_input)[0]
        probability = best_model.predict_proba(X_input)[0]
        
        return {
            'prediction': self.encoders['Depression'].inverse_transform([prediction])[0],
            'confidence': max(probability),
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.4 else 'Low',
            'probabilities': {
                'No Depression': probability[0],
                'Depression': probability[1]
            },
            'best_model_used': self.best_model
        }
    
    def load_models(self):
        try:
            data = joblib.load('all_models.pkl')
            self.models = data['models']
            self.encoders = data['encoders']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.best_model = data['best_model']
            self.performance_results = data['performance_results']
            return True
        except:
            return False
