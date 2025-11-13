from fastapi import FastAPI, UploadFile, File
import whisper
import uvicorn
import os
import uuid

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("base.en")

@app.get("/")
def home():
    return {"message": "Whisper API is running!"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = model.transcribe(file_path)

    return {
        "text": result["text"],
        "saved_file": file_path
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)

