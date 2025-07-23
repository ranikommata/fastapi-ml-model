from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app=FastAPI()
model_path=os.path.join(os.path.dirname(__file__),'model.pkl')
model=joblib.load(model_path)
class InputData(BaseModel):
    marks:list[float]
@app.post("/predict")  #localhost:5000/docs
def predict(data:InputData):
    #return "testing"
    prediction=model.predict([data.marks])
    return {"prediction":int(prediction[0])}