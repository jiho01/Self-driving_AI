# OpenCV란?

**OpenCV (Open Source Computer Vision Library)**는 오픈소스 컴퓨터 비전 라이브러리로, 이미지 처리, 영상 분석, 객체 인식, 머신 러닝 등 다양한 기능을 제공합니다.

- 개발 언어: Python, C++, Java 등
- 주요 특징: 빠른 처리 속도, 실시간 처리 지원
- 사용 분야: 얼굴 인식, 자율주행, 필터 앱, OCR, 모션 추적 등

---

# 🛠️ 설치 방법

Python 환경에서 pip로 쉽게 설치할 수 있습니다.

```bash
# 일반 GUI 환경용
pip install opencv-python

# GUI 창이 필요 없는 서버 환경 (headless)
pip install opencv-python-headless
```

설치 확인:
```python
import cv2
print(cv2.__version__)
```

---

# 📚 기본 용어 정리

| 용어 | 설명 |
|------|------|
| **BGR** | OpenCV가 사용하는 기본 색상 순서 (Blue, Green, Red) |
| **RGB** | 일반적으로 사용하는 색상 순서 (Red, Green, Blue) |
| **Grayscale** | 흑백 이미지 (밝기 정보만 포함) |
| **Pixel** | 이미지의 한 점. (x, y) 좌표와 색상 정보를 가짐 |
| **Frame** | 영상(비디오)의 한 장면 (정지 이미지) |
| **Cascade Classifier** | 얼굴 탐지 등에서 사용하는 학습된 객체 인식 모델 |
| **Kernel** | 이미지 필터링 시 사용하는 행렬 (ex: Blur, Edge 등) |

---

# 🔧 주요 기능 요약

| 기능 | 설명 |
|------|------|
| 이미지 읽기/쓰기 | `cv2.imread()`, `cv2.imwrite()` |
| 이미지 출력 | `cv2.imshow()` |
| 색상 변환 | `cv2.cvtColor()` |
| 이미지 자르기 | 배열 슬라이싱 (`img[y1:y2, x1:x2]`) |
| 블러 처리 | `cv2.GaussianBlur()`, `cv2.blur()` |
| 에지 검출 | `cv2.Canny()` |
| 객체 탐지 | `cv2.CascadeClassifier()` |
| 웹캠 영상 처리 | `cv2.VideoCapture()` |

---

# 📸 기본 예제 모음

## 1. 이미지 불러오기 & 보기
```python
import cv2

img = cv2.imread('image.jpg')           # 이미지 불러오기
cv2.imshow('My Image', img)             # 이미지 창에 띄우기
cv2.waitKey(0)                          # 키 입력 대기
cv2.destroyAllWindows()                 # 창 닫기
```

## 2. 이미지 흑백 변환
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
```

## 3. 웹캠 영상 출력
```python
cap = cv2.VideoCapture(0)  # 0 = 기본 카메라

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 4. 얼굴 인식 (Haar Cascade 사용)
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

# 🔗 참고 자료

- OpenCV 공식 문서: https://docs.opencv.org/
- 튜토리얼: https://opencv-python-tutroals.readthedocs.io/
- GitHub: https://github.com/opencv/opencv

---

> 💡 참고: OpenCV는 실시간 처리에 강하므로, 영상 기반 프로젝트에 적합합니다.
