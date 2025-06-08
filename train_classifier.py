import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Veriyi oku
df = pd.read_csv("Marmara.csv")

# Sınıf etiketleri üret
def classify_magnitude(mag):
    if mag < 3.0:
        return "Hafif"
    elif mag < 5.0:
        return "Orta"
    else:
        return "Şiddetli"

df["Magnitude_Class"] = df["Magnitude_ML"].apply(classify_magnitude)

# 8 özellik (Mw_Potential yok!)
features = ["Depth_km", "Latitude", "Longitude", "Fault_Length_km", "Fault_Width_m",
            "Slip_Rate_mm_per_yr", "Elapsed_Time_yr", "Slip_Deficit_m"]

df = df.dropna(subset=features + ["Magnitude_Class"])

X = df[features]
y = df["Magnitude_Class"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "classifier_model_v2.pkl")
joblib.dump(le, "label_encoder_v2.pkl")

print("✅ 8 özellikli RandomForest modeli başarıyla kaydedildi.")
