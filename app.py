from flask import Flask, render_template, request
import pickle, joblib, pandas as pd, requests

app = Flask(__name__)

# Load trained models & encoders
MODEL_DIR = "models/"

xgb = joblib.load(f"{MODEL_DIR}crop_xgb.joblib")
xgb_f = joblib.load(f"{MODEL_DIR}xgb_fertilizer.joblib")

le = joblib.load(f"{MODEL_DIR}le.pkl")
le_soil = pickle.load(open(f"{MODEL_DIR}le_soil.pkl", "rb"))
le_crop = pickle.load(open(f"{MODEL_DIR}le_crop.pkl", "rb"))
le_fert = pickle.load(open(f"{MODEL_DIR}le_fert.pkl", "rb"))

print("Models and encoders loaded successfully!")


#  Weather API Function

def get_weather(city_name):
    api_key = "c3ceb51fd78d5abddbf044e949db9337" 
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": api_key, "units": "metric"}

    response = requests.get(base_url, params=params)
    data = response.json()

    if data.get("cod") != 200:
        return None

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rainfall = data.get("rain", {}).get("1h", 0)
    return temperature, humidity, rainfall

#Crop & Fertilizer Prediction
def predict_crop_and_fertilizer(N, P, K, temperature, humidity, ph, rainfall, soil_type, city):
    crop_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    probs = xgb.predict_proba(crop_input)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3_crops = le.inverse_transform(top3_idx)
    crop_prediction = top3_crops[0]
    confidence = round(probs[top3_idx[0]] * 100, 2)

    # Fertilizer prediction
    soil_type = soil_type.strip().capitalize()
    try:
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded_fert = le_crop.transform([crop_prediction])[0]
        fert_input = pd.DataFrame([[temperature, humidity, 50, N, P, K, soil_encoded, crop_encoded_fert]],
                                  columns=['Temparature', 'Humidity ', 'Moisture', 'Nitrogen',
                                           'Phosphorous', 'Potassium', 'Soil_Type_enc', 'Crop_Type_enc'])
        fert_encoded = xgb_f.predict(fert_input)[0]
        fert_prediction = le_fert.inverse_transform([fert_encoded])[0]
    except:
        fert_prediction = "NPK 20-20-20"

    return crop_prediction, fert_prediction, confidence, top3_crops[1], top3_crops[2]


# Soil Optimization Suggestions
def get_soil_tip(N, P, K, ph):
    tips = []

    # Nitrogen
    if N < 40:
        tips.append("🧪 Low Nitrogen: Apply Urea or compost to boost leaf growth.")
    elif N > 120:
        tips.append("⚠️ High Nitrogen: Reduce urea use to avoid excessive leaf growth.")

    # Phosphorus
    if P < 30:
        tips.append("🌾 Low Phosphorus: Use DAP or bone meal for better root strength.")
    elif P > 100:
        tips.append("⚠️ High Phosphorus: Limit DAP use; excess P reduces zinc uptake.")

    # Potassium
    if K < 30:
        tips.append("🌿 Low Potassium: Add MOP fertilizer to improve drought resistance.")
    elif K > 80:
        tips.append("⚠️ High Potassium: Avoid MOP; too much can hinder calcium absorption.")

    # pH value
    if ph < 5.5:
        tips.append("🍋 Acidic Soil: Apply lime to raise pH for optimal crop growth.")
    elif ph > 8:
        tips.append("🧂 Alkaline Soil: Add gypsum or organic matter to lower soil pH.")

    if not tips:
        tips.append("✅ Soil nutrient levels are balanced — no major adjustments needed.")

    return tips



# Flask Routes

@app.route('/')
def home():
    return render_template('index.html')

from flask import redirect

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return redirect('/') 
    try:
        city = request.form['city']
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        ph = float(request.form['ph'])
        soil_type = request.form['soil_type']

        
        weather = get_weather(city)
        if not weather:
            return render_template('index.html', error=f"⚠️ Weather data not found for {city}. Check spelling!")

        temperature, humidity, rainfall = weather

       
        crop, fertilizer, confidence, alt1, alt2 = predict_crop_and_fertilizer(
            N, P, K, temperature, humidity, ph, rainfall, soil_type, city
        )

        
        soil_tips = get_soil_tip(N, P, K, ph)

        return render_template('index.html', crop=crop, fertilizer=fertilizer, city=city,
                               temperature=temperature, humidity=humidity, rainfall=rainfall,
                               confidence=confidence, alt1=alt1, alt2=alt2, soil_tips=soil_tips)

    except Exception as e:
        return render_template('index.html', error=str(e))



if __name__ == '__main__':
    print("🚀 Starting AgriVision Flask server...")
    app.run(debug=True)
