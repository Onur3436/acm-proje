import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Veriyi oku
df = pd.read_csv("Marmara.csv")

# Kullanılacak özellikler
features = [
    "Depth_km", "Latitude", "Longitude", "Fault_Length_km",
    "Fault_Width_m", "Slip_Rate_mm_per_yr", "Elapsed_Time_yr", "Slip_Deficit_m"
]

# Eksik verileri temizle
df = df.dropna(subset=features + ["Mw_Potential"])

# X: Girdi, y: Risk skoru (örnek formül)
X = df[features]
df["Risk_Score"] = df["Mw_Potential"] * df["Elapsed_Time_yr"]
y = df["Risk_Score"]

# Eğitim ve test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model eğitimi
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(model, "risk_model.pkl")
print("✅ Risk skoru modeli başarıyla kaydedildi: risk_model.pkl")
