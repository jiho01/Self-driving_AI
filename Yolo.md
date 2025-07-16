# YOLOv8을 이용한 객체 탐지 프로젝트

## 📖 프로젝트 개요

이 프로젝트는 Ultralytics의 YOLOv8 모델을 사용하여 실시간 객체 탐지를 수행하는 것을 목표로 합니다. COCO 데이터셋으로 사전 훈련된 모델을 사용하여 다양한 객체를 탐지하고, 커스텀 데이터셋을 이용한 모델 학습 및 평가를 지원합니다.

## ✨ 주요 기능

- **실시간 객체 탐지**: 비디오 및 이미지에서 실시간으로 객체를 탐지합니다.
- **모델 학습**: 커스텀 데이터셋을 사용하여 YOLOv8 모델을 재학습할 수 있습니다.
- **모델 평가**: 학습된 모델의 성능을 평가하고 mAP, F1-score 등의 지표를 확인합니다.
- **다양한 모델 지원**: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x 등 다양한 크기의 모델을 지원합니다.

## 🛠️ 설치 방법

1. **Git 리포지토리 클론**

   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **필요한 라이브러리 설치**

   이 프로젝트는 다음 라이브러리를 사용합니다. `requirements.txt` 파일을 통해 한 번에 설치할 수 있습니다.

   ```bash
   pip install -r requirements.txt
   ```

   **`requirements.txt` 내용:**
   ```
   ultralytics
   tensorflow
   matplotlib
   ```

   * `ultralytics`: YOLOv8 모델을 사용하기 위한 필수 라이브러리입니다.
   * `tensorflow`: (선택 사항) 다른 딥러닝 작업을 위해 포함되었습니다.
   * `matplotlib`: 결과 시각화를 위해 사용됩니다.

## 🚀 사용 방법

### 1. 사전 훈련된 모델을 이용한 추론

- **이미지 추론:**
  ```bash
  yolo predict model=yolov8n.pt source='path/to/your/image.jpg'
  ```

- **비디오 추론:**
  ```bash
  yolo predict model=yolov8n.pt source='path/to/your/video.mp4'
  ```

### 2. 커스텀 모델 학습

1. **데이터 준비**: `coco8.yaml`과 같은 형식으로 데이터셋 설정 파일을 준비합니다.

   ```yaml
   # coco8.yaml 예시
   path: ../datasets/coco8  # 데이터셋 루트 디렉토리
   train: images/train  # 학습 이미지 경로 (path에 상대적)
   val: images/val    # 검증 이미지 경로 (path에 상대적)

   # 클래스 이름
   names:
     0: person
     1: bicycle
     2: car
     # ...
   ```

2. **모델 학습 실행**:

   ```bash
   yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640
   ```

   - `model`: 베이스 모델 (e.g., `yolov8n.pt`)
   - `data`: 데이터셋 설정 파일
   - `epochs`: 학습 에포크 수
   - `imgsz`: 입력 이미지 크기

### 3. 학습된 모델 평가

학습이 완료되면 `runs/train/exp/weights/best.pt` 와 같은 경로에 최적의 가중치 파일이 저장됩니다. 이 모델을 사용하여 성능을 평가할 수 있습니다.

```bash
yolo val model='path/to/your/best.pt' data=coco8.yaml
```

## 📈 결과

학습 및 추론 결과는 `runs` 디렉토리 내에 자동으로 저장됩니다. 각 실행마다 `exp`, `exp2`, ... 와 같은 폴더가 생성되며, 내부에는 다음 정보가 포함됩니다.

- **가중치 파일 (`weights/`)**: `best.pt` (최고 성능 모델), `last.pt` (마지막 모델)
- **결과 그래프 (`results.png`)**: 손실 및 성능 지표 그래프
- **혼동 행렬 (`confusion_matrix.png`)**: 클래스별 예측 성능
- **예측 결과 예시**: 검증 데이터셋에 대한 예측 결과 이미지

## 🤝 기여하기

이 프로젝트에 기여하고 싶으시다면 언제든지 Pull Request를 보내주시거나 이슈를 등록해주세요.

---

*이 README 파일은 Gemini에 의해 생성되었습니다.*
