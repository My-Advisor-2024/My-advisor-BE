from fastapi import FastAPI , Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from handler import preprocess_image, preprocess_metadata, predict_with_model

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000", 
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
    try:
        # 1. 이미지 전처리
        input_image = preprocess_image(photo.file)

        # 2. 메타데이터 전처리
        input_meta = preprocess_metadata(gender, age, location)

        # 3. 모델 예측
        prediction_result = predict_with_model(input_image, input_meta)

        # 4. 결과 반환
        return JSONResponse(
            content=prediction_result,
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )