from fastapi import FastAPI, UploadFile, File   #(api framework)
from fastapi.staticfiles import StaticFiles   
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = tf.keras.models.load_model("Cricket_Shot_Detection_model.keras")
@app.get("/")
def home():
    return FileResponse("static/index.html")
@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB").resize((224,224))
    img_arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prediction = model.predict(img_arr)[0]
    idx = int(np.argmax(prediction))
    CLASS_NAMES = ["drive", "legglance-flick", "pullshot", "sweep"]

    Display_names = {"drive": "DRIVE",
                     "legglance-flick":"FLICK",
                     "pullshot": "PULL",
                     "sweep":"SWEEP"}
    
    return {
        "shot":Display_names[CLASS_NAMES[idx]] , 
        "confidence":round(float(prediction[idx]) * 100,2),
        "All_probabilities":{
            Display_names[name]: round(float(p)*100, 2)
            for name,p in zip(CLASS_NAMES, prediction)
            }
        }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    