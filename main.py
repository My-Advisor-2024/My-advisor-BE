from PIL import Image
from fastapi import FastAPI , Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:5500", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 도메인 리스트
    allow_credentials=True,  # 쿠키를 포함한 인증 정보 허용
    allow_methods=["*"],     # 모든 HTTP 메서드 허용 (GET, POST, PUT 등)
    allow_headers=["*"],     # 모든 HTTP 헤더 허용
)

@app.get("/")
async def root():
    return {"message": "Welcome to the My advisor"}

@app.post("/predict")
async def predict(
    gender: str = Form(...),
    age: int = Form(...),
    location: str = Form(...),
    photo: UploadFile = File(...)
):
    # 이미지 검증
    try:
        img = Image.open(photo.file)
        img.verify()  # 이미지가 유효한지 확인
    except Exception:
        return JSONResponse(
            content={"error": "Invalid image file"}, status_code=400
        )
    
    # Mock AI Prediction
    prediction = f"Predicted condition for {location} is Skin Rash"

    return JSONResponse(
        content={"prediction": prediction}, status_code=200
    )