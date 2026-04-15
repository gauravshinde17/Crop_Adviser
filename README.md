# 🌾 Crop Recommendation System

A machine learning web application that recommends the most suitable crop to cultivate based on soil composition and environmental conditions. Built during a Data Science internship at CODTECH IT Solutions.

---

## Problem Statement

Farmers often select crops based on intuition or tradition rather than data — leading to suboptimal yields and resource waste. This system provides a data-driven recommendation by analyzing soil nutrients and climate parameters, helping farmers make informed cultivation decisions.

---

## What I Built

A Flask-based web application where users input 7 agronomic parameters and receive an instant crop recommendation from a trained ML classification model.

**Input Parameters:**
| Parameter | Description | Range |
|---|---|---|
| Nitrogen (N) | Soil nitrogen content | 0 – 100 |
| Phosphorus (P) | Soil phosphorus content | 0 – 100 |
| Potassium (K) | Soil potassium content | 0 – 100 |
| Temperature | Ambient temperature (°C) | 0 – 50 |
| Humidity | Relative humidity (%) | 0 – 100 |
| Soil pH | Soil acidity/alkalinity | 4.0 – 8.0 |
| Rainfall | Annual rainfall (mm) | 0 – 3000 |

**Output:** Single best-fit crop recommendation across 22 crop classes:
> Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidney Beans, Lentil, Maize, Mango, Moth Beans, Mung Bean, Muskmelon, Orange, Papaya, Pigeon Peas, Pomegranate, Rice, Watermelon

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3 |
| Backend | Python, Flask |
| ML Model | Scikit-learn — RandomForestClassifier (100 trees, random_state=42) |
| Model Serialization | Pickle |
| Dataset | Crop Recommendation Dataset (Kaggle) |

---

## Application Architecture

```
User Input (HTML Form)
        ↓
Flask Backend (/predict endpoint)
        ↓
Pickle Model (trained_model.pkl)
        ↓
Predicted Crop → JSON Response → UI Display
```

---

## Project Structure

```
Crop_Adviser/
│
├── Crop-Advisor/
│   ├── app.py                  # Flask application & prediction route
│   ├── models/
│   │   └── trained_model.pkl   # Serialized ML model
│   └── templates/
│       └── index.html          # Web UI
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/gauravshinde17/Crop_Adviser.git
cd Crop_Adviser/Crop-Advisor

# 2. Install dependencies
pip install flask scikit-learn pandas numpy

# 3. Run the app
python app.py

# 4. Open in browser
http://localhost:5000
```

---

## Key Learnings

- Built and serialized a multi-class ML classification model using scikit-learn
- Developed a REST API using Flask to serve real-time predictions
- Connected a frontend form to a Python backend via `fetch()` API calls
- Applied feature engineering on agronomic datasets for model training

---

## Internship Context

Developed as part of the **Data Science Internship at CODTECH IT Solutions Pvt. Ltd.** (Jan 2025 – Feb 2025)
