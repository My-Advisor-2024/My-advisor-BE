import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# 모델 및 전처리 도구 로드
model_path = 'assets/class_weight_model.h5'
encoder_path = 'assets/encoder.pkl'
scaler_path = 'assets/scaler.pkl'
label_encoder_path = 'assets/label_encoder.pkl'

# 모델, 인코더, 스케일러 로드
model = load_model(model_path)
encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# 모델 컴파일 (경고 제거)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image_file, target_size=(128, 128)):
    """
    이미지를 읽고 모델 입력 크기로 전처리.
    """
    try:
        # 이미지를 PIL로 열고, OpenCV 형식으로 변환
        img = Image.open(image_file)
        img = img.convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, target_size) / 255.0  # 크기 조정 및 정규화
        return np.expand_dims(img, axis=0)  # 배치 차원 추가
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {e}")


def preprocess_metadata(sex: str, age: int, localization: str):
    """
    성별, 나이, 병변 위치 등의 메타데이터를 전처리.
    """
    try:
        # 메타데이터를 DataFrame으로 변환
        meta_data = pd.DataFrame([[sex, localization]], columns=['sex', 'localization'])
        age_data = pd.DataFrame([[age]], columns=['age'])

        # OneHotEncoder 및 MinMaxScaler 적용
        encoded_features = encoder.transform(meta_data)
        age_normalized = scaler.transform(age_data)

        # 메타데이터 병합
        input_meta = np.hstack((age_normalized, encoded_features)).reshape(1, -1)
        return input_meta
    except Exception as e:
        raise ValueError(f"Metadata preprocessing error: {e}")


def predict_with_model(input_image: np.ndarray, input_meta: np.ndarray):
    """
    모델에 입력 데이터를 제공하여 예측 수행.
    """
    try:
        # 모델 예측
        predictions = model.predict([input_image, input_meta])  # 모델에 이미지와 메타데이터 입력
        predicted_class = np.argmax(predictions, axis=1)  # 가장 높은 확률의 클래스 선택

        # 결과 반환
        predicted_label = label_encoder.classes_[predicted_class[0]]
        class_probabilities = predictions.tolist()  # 확률값을 JSON 직렬화 가능하게 변환

        return {
            "prediction": predicted_label,
            "class_probabilities": class_probabilities[0],
        }
    except Exception as e:
        raise ValueError(f"Model prediction error: {e}")