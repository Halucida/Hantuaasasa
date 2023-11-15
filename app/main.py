from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Model API")

@app.get("/")
def root():
    return {"msg":"Hii !"}

@app.post("/predict")
def model_pinpoint(image : ):
    result = model.pinpoint(image)
    return result