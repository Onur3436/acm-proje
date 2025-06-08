import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("Marmara.csv")

# Özellikler (girdiler) → Mw_Potential yok!
features = ["Depth_km", "Latitude", "Longitude", "Fault_Length_km", "Fault_Width_m",
            "Slip_Rate_mm_per_yr", "Elapsed_Time_yr", "Slip_Deficit_m"]
target = "Mw_Potential"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Yeni model kaydedilir
joblib.dump(model, "regression_model.pkl")

print("✅ Regressor modeli başarıyla yeniden eğitildi.")
