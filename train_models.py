import model_training as mt
import pandas as pd

if __name__ == "__main__":
    model = mt.AdvancedStudentHealthModel()
    # Read CSV and strip spaces from columns
    df = pd.read_csv("student_data.csv")  # <- your actual CSV
    df.columns = df.columns.str.strip()
    df.to_csv("student_data.csv", index=False)

    print("Columns found:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    # Train models
    results = model.train_all_models("student_data.csv")
    print("Training complete. Model performances:")
    for name, info in results.items():
        print(f"{name}: Accuracy = {info['accuracy']:.4f}")
