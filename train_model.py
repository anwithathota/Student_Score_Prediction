import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


# -----------------------------------
# 1. Load the dataset
# -----------------------------------

data = pd.read_csv("dataset/student_data.csv")

print("Dataset loaded successfully!")

print(data)


# -----------------------------------
# 2. Separate input and output
# -----------------------------------

X = data[
    [
        "Hours_Studied",
        "Previous_Score",
        "Attendance",
        "Sleep_Hours"
    ]
]

y = data["Final_Score"]


# -----------------------------------
# 3. Split the dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))


# -----------------------------------
# 4. Create Linear Regression model
# -----------------------------------

model = LinearRegression()


# -----------------------------------
# 5. Train the model
# -----------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# -----------------------------------
# 6. Make predictions
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# 7. Calculate performance
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nModel Performance")
print("-------------------------")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)


# -----------------------------------
# 8. Create model folder
# -----------------------------------

os.makedirs("model", exist_ok=True)


# -----------------------------------
# 9. Save the trained model
# -----------------------------------

joblib.dump(
    model,
    "model/student_score_model.pkl"
)

print("\nModel saved successfully!")

print("Location: model/student_score_model.pkl")