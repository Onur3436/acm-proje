from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import requests

app = Flask(__name__)

# Modelleri yükle
classifier = joblib.load("classifier_model_v2.pkl")
regressor = joblib.load("regression_model.pkl")
label_encoder = joblib.load("label_encoder_v2.pkl")

# Veri seti
df = pd.read_csv("Marmara.csv")
df["Location"] = df["Location"].astype(str).str.strip().str.title()

# Risk skoru hesapla
def compute_risk_score(row):
    elapsed = row["Elapsed_Time_yr"]
    slip_deficit = row["Slip_Deficit_m"]
    slip_rate = row["Slip_Rate_mm_per_yr"]
    elapsed_norm = elapsed / 100
    slip_deficit_norm = slip_deficit / 5
    slip_rate_norm = slip_rate / 50
    score = (elapsed_norm + slip_deficit_norm + slip_rate_norm) / 3 * 100
    return min(round(score, 2), 100)

df["Risk_Score"] = df.apply(compute_risk_score, axis=1)

# En yakın lokasyonu bul
def find_nearest_location(lat, lon, threshold_km=10):
    min_dist = float("inf")
    nearest_row = None
    for _, row in df.iterrows():
        loc = (row["Latitude"], row["Longitude"])
        dist = geodesic((lat, lon), loc).km
        if dist < min_dist:
            min_dist = dist
            nearest_row = row
    if min_dist <= threshold_km:
        return nearest_row
    else:
        return None

# LLM'den daha net ve bağlamsal cevap al
def ask_llm(question, context_info):
    try:
        prompt = f"""
Model çıktısı:
{context_info}

Kullanıcının sorusu:
{question}

Yalnızca yukarıdaki çıktıya dayanarak, kısa ve açık bir Türkçe ile net bir cevap ver. Etikette belirtilen risk düzeyiyle çelişen yorumlar yapma. Cevap en fazla 2 cümle olsun. Yapay, ezber cümleler ya da uydurma açıklamalar yazma.

Cevap:
"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": "dolphin-mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"LLM hatası: {e}"


@app.route("/")
def home():
    return render_template("map.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])

        nearest = find_nearest_location(lat, lon)
        if nearest is None:
            return jsonify({"status": "error", "message": "Bu bölgede deprem verisi bulunmamaktadır."})

        features = [
            nearest["Depth_km"],
            nearest["Latitude"],
            nearest["Longitude"],
            nearest["Fault_Length_km"],
            nearest["Fault_Width_m"],
            nearest["Slip_Rate_mm_per_yr"],
            nearest["Elapsed_Time_yr"],
            nearest["Slip_Deficit_m"]
        ]
        input_data = np.array([features])
        class_pred = classifier.predict(input_data)
        label = label_encoder.inverse_transform(class_pred)[0]
        magnitude = round(regressor.predict(input_data)[0], 2)
        risk_score = round(nearest["Risk_Score"], 2)

        # Model çıktısı açıklaması (LLM için kullanılacak bağlam)
        context = (
            f"{nearest['Location']} bölgesi için deprem büyüklüğü tahmini: {magnitude}, "
            f"etiket: {label}, risk skoru: {risk_score}."
        )

        return jsonify({
            "status": "success",
            "city": nearest["Location"],
            "label": label,
            "magnitude": magnitude,
            "risk_score": risk_score,
            "context": context
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.form["question"]
        context = request.form["context"]
        answer = ask_llm(question, context)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"LLM hatası: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
