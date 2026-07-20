import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

n = int(input("Enter number of student records: "))

study_hours = []
attendance = []
assignment = []
final_marks = []

for i in range(n):
    print(f"\nStudent {i+1}")

    study_hours.append(float(input("Study Hours: ")))
    attendance.append(float(input("Attendance : ")))
    assignment.append(float(input("Assignment Score: ")))
    final_marks.append(float(input("Final Marks: ")))


df = pd.DataFrame({
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Assignment": assignment,
    "Final Marks": final_marks
})

df.to_csv("Student_Data.csv", index=False)

X = df[["Study Hours", "Attendance", "Assignment"]]
y = df["Final Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nRegression Equation:")

print(
    f"Y = {model.intercept_:.2f} + "
    f"{model.coef_[0]:.2f}X1 + "
    f"{model.coef_[1]:.2f}X2 + "
    f"{model.coef_[2]:.2f}X3"
)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

if len(y_test) > 1:
    r2 = r2_score(y_test, y_pred)
else:
    r2 = 0

print("\nMODEL PERFORMANCE")

print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

plot_df = df.sort_values("Study Hours")

plt.figure(figsize=(7,5))

plt.plot(
    plot_df["Study Hours"],
    plot_df["Final Marks"],
    '-',
    color="#f43f5e",
    linewidth=2
)

plt.scatter(
    plot_df["Study Hours"],
    plot_df["Final Marks"],
    color="blue",
    s=70
)

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()


metrics_names = ["MSE", "MAE", "RMSE", "R²"]
metrics_vals = [mse, mae, rmse, r2]

colors = ["#a855f7", "#08d6d4", "#f43f5e", "#10b981"]

plt.figure(figsize=(7,5))

bars = plt.bar(
    metrics_names,
    metrics_vals,
    color=colors,
    width=0.5
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.title("Model Performance Metrics")
plt.ylabel("Metric Value")
plt.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.show()