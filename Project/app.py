from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np
import traceback
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and features
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "house_price_model.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))
    print("✅ System Ready.")
except Exception as e:
    print(f"❌ Load Error: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "form": {
                "area": "", "bedrooms": 2, "bathrooms": 1, "stories": 1,
                "mainroad": "yes", "guestroom": "no", "basement": "no",
                "hotwaterheating": "no", "airconditioning": "no", "parking": 0,
                "prefarea": "no", "furnishingstatus": "semi-furnished"
            }
        }
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    stories: int = Form(...),
    mainroad: str = Form(...),
    guestroom: str = Form(...),
    basement: str = Form(...),
    hotwaterheating: str = Form(...),
    airconditioning: str = Form(...),
    parking: int = Form(...),
    prefarea: str = Form(...),
    furnishingstatus: str = Form(...)
):
    form_data = {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "stories": stories, "mainroad": mainroad, "guestroom": guestroom,
        "basement": basement, "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning, "parking": parking,
        "prefarea": prefarea, "furnishingstatus": furnishingstatus
    }
    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame([form_data])

        # 2. Ensure column order
        df = df[features]

        # 3. Predict log price
        pred_log = model.predict(df)

        # 4. Invert log transformation
        price = np.expm1(pred_log[0])

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": f"{round(float(price), 2):,}",
                "form": form_data
            }
        )

    except Exception:
        error_details = traceback.format_exc()
        print(f"❌ PREDICTION ERROR:\n{error_details}")
        return HTMLResponse(content=f"<h3>Error: {error_details}</h3>", status_code=500)


