#1-Import Libraries
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

#3-Create the app object
app=FastAPI()
pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#4-Index route-opnes automaically on http://127.0.0.1:8000
@app.get("/")
def index():
    return{"Message":"Hello Stranger!"}
#5-Routing with a single parameter
@app.get("/{name}")
def get_name(name:str):
    return {"message":f"Hello,{name}"}

#6-Expose the prediction functionality-make a prediction from the passed JSON data and return the prediction with confidence
@app.post("/predict") #API name
def pred_bank(data:BankNote): #capture the i/p features needed from the class object and assign to an object data
    data=data.dict()
    print(data)
    variance=data["variance"]
    print(variance)
    skewness=data["skewness"]
    curtosis=data['curtosis']
    entropy=data["entropy"]
    print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    if prediction[0]>0.5:
        prediction="Fake_Note"
    else:
        prediction="Genuine Note"
    return {"prediction":prediction}

#7-Run the API with uvicorn
#It will run on http://127.0.0.1:8000
if __name__=="__main__":
    uvicorn.run(app,host='127.0.0.1',port=8000)

#8-use uvicorn app:app --reload command at the anaconda cmd prompt




