# 🏠 House Price Prediction & Web Application

An end-to-end Machine Learning project to predict residential property prices in Indian Rupees (₹) using supervised regression models. Built with **Scikit-learn**, **FastAPI**, and an interactive web interface.

---

## 📌 Project Overview

This repository provides:
1. **Exploratory Data Analysis & Model Training** (`cookbook.ipynb`, `train_and_export.py`): Full pipeline analyzing housing factors, feature engineering, and model selection.
2. **Tuned ML Pipeline**: A Scikit-learn pipeline utilizing `StandardScaler` for continuous numerical features, `OneHotEncoder` for categorical attributes, and **Ridge Regression ($\alpha=10$)** achieving an $R^2$ score of ~0.68.
3. **Interactive Web App** (`Project/`): A clean, responsive FastAPI + Jinja2 frontend allowing users to input property specifications and receive instant price valuations.
4. **REST API** (`app.py`): A standalone JSON endpoint (`POST /predict`) for programmatic integration.

---

## 📊 Dataset & Features

The model is trained on [`Housing.csv`](./Housing.csv) containing **545 property records** and **13 attributes**:

* **Target Variable**: `price` (log-transformed using `np.log1p` during training to normalize distribution).
* **Numerical Features**:
  * `area` (in square feet)
  * `bedrooms`, `bathrooms`, `stories`, `parking`
* **Categorical / Utility Features**:
  * `mainroad` (`yes`/`no`)
  * `guestroom` (`yes`/`no`)
  * `basement` (`yes`/`no`)
  * `hotwaterheating` (`yes`/`no`)
  * `airconditioning` (`yes`/`no`)
  * `prefarea` (`yes`/`no`)
  * `furnishingstatus` (`furnished`, `semi-furnished`, `unfurnished`)

---

## 🏆 Model Benchmarking

Multiple regression algorithms were evaluated and cross-validated:

| Model | $R^2$ Score | RMSE (log scale) | RMSE (₹) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.6722 | 0.2515 | ~₹1,314,648 |
| **Ridge Regression ($\alpha=10$)** | **0.6789 (CV)** | **0.2518** | **~₹1,315,541** |
| **Gradient Boosting** | 0.6708 | 0.2521 | ~₹1,336,992 |
| **Random Forest** | 0.6300 | 0.2673 | ~₹1,432,202 |
| **SVR** | 0.6223 | 0.2701 | ~₹1,424,178 |
| **KNN** | 0.5611 | 0.2911 | ~₹1,565,776 |
| **Decision Tree** | 0.5097 | 0.3077 | ~₹1,631,237 |

---

## 🛠️ Project Structure

```bash
├── app.py                      # Standalone FastAPI REST API
├── train_and_export.py         # Pipeline training & pickle export script
├── cookbook.ipynb              # Jupyter notebook with EDA & experiments
├── Housing.csv                 # Dataset
├── house_price_model.pkl       # Fitted Scikit-learn Pipeline artifact
├── model_features.pkl          # Saved feature column ordering
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── Project/
    ├── app.py                  # FastAPI server with Jinja2 template routes
    ├── house_price_model.pkl   # Fitted model for Web App
    ├── model_features.pkl      # Feature definitions for Web App
    └── templates/
        └── index.html          # Interactive prediction UI
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/aanuvesh24/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Retrain and Export Model
```bash
python train_and_export.py
```

### 4. Run the Web Application
```bash
cd Project
uvicorn app:app --reload --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🌐 API Usage

You can also send a POST request to the REST API:

```bash
uvicorn app:app --port 8000
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "area": 7420,
       "bedrooms": 4,
       "bathrooms": 2,
       "stories": 3,
       "mainroad": "yes",
       "guestroom": "no",
       "basement": "no",
       "hotwaterheating": "no",
       "airconditioning": "yes",
       "parking": 2,
       "prefarea": "yes",
       "furnishingstatus": "furnished"
     }'
```

**Response:**
```json
{
  "predicted_price": 8343391.11
}
```
