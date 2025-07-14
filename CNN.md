
# 🧠 Convolutional Neural Network (CNN) 및 이미지 처리·표시 방법

CNN(Convolutional Neural Network)은 이미지나 영상 같은 시각 데이터를 처리하기 위해 설계된 딥러닝 모델입니다. 주로 **이미지 분류, 객체 인식, 자율주행, 얼굴 인식** 등에 사용됩니다.

---

## 🔧 주요 구성 요소 및 용어 정리

### 1. Convolution Layer (합성곱 층)
- 필터(커널)를 사용하여 입력 이미지에서 **특징(Feature)** 을 추출합니다.
- 입력 이미지의 국소 영역에 필터를 곱하여 특징 맵(feature map)을 생성합니다.
- 여러 개의 필터를 사용하면 다양한 특징을 학습할 수 있습니다.

```python
# 예시 (PyTorch)
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

---

### 2. ReLU (Rectified Linear Unit)
- 비선형 활성화 함수로, 모델에 **비선형성**을 부여합니다.
- 음수는 0으로, 양수는 그대로 유지합니다.

```python
# 예시
F.relu(x)
```

---

### 3. Pooling Layer (풀링층)
- 특징 맵의 크기를 줄여 연산량을 줄이고, 특징의 위치 변화에 대해 **불변성**을 제공합니다.
- 종류: Max Pooling, Average Pooling 등

```python
# 예시 (Max Pooling)
nn.MaxPool2d(kernel_size=2, stride=2)
```

---

### 4. Fully Connected Layer (완전 연결층)
- 마지막 단계에서 추출된 특징을 기반으로 **분류(Classification)** 를 수행합니다.
- 일반적인 인공신경망(Dense Layer)와 동일한 구조입니다.

```python
# 예시
nn.Linear(in_features=128, out_features=10)
```

---

## 🔄 전체 흐름 예시

```
입력 이미지
   ↓
[Convolution → ReLU → Pooling] × n
   ↓
Flatten (1차원으로 변환)
   ↓
Fully Connected Layer
   ↓
Softmax (출력 확률)
   ↓
클래스 예측 (예: 고양이 vs 강아지)
```

---

## 🎯 CNN의 특징

- 이미지의 **2차원 구조를 그대로 유지**하며 처리
- 파라미터 수가 적고 계산 효율이 높음
- 특징을 **자동으로 추출**함 (수작업 특징 설계 필요 없음)
- 영상 처리, 의료 영상, 자율주행, 얼굴 인식 등에서 폭넓게 활용됨

---

## 📚 주요 활용 분야

| 분야        | 예시                       |
|-------------|----------------------------|
| 이미지 분류 | 고양이 vs 강아지, 숫자 인식 |
| 객체 탐지   | 자율주행 자동차, CCTV 분석 |
| 얼굴 인식   | Face ID, 감정 분석         |
| 의료 영상   | CT/MRI 이상 탐지           |

---

## 🖼️ 이미지 표시 방법

아래는 로컬 환경 및 Jupyter/Colab 등에서 이미지를 화면에 띄우는 세 가지 방법입니다.

### 1. Matplotlib 사용하기
```python
import cv2
from matplotlib import pyplot as plt

# 이미지 읽기 (BGR)
img = cv2.imread('내이미지.jpg')
# BGR → RGB 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
plt.title('My Image')
plt.show()
```

### 2. OpenCV 자체 윈도우 (로컬 스크립트)
```python
import cv2

img = cv2.imread('내이미지.jpg')
if img is None:
    print("이미지 경로를 확인하세요")
else:
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 3. Colab / Jupyter 환경에서 cv2_imshow 사용
```python
from google.colab.patches import cv2_imshow
import cv2

img = cv2.imread('내이미지.jpg', cv2.IMREAD_COLOR)
cv2_imshow(img)
```

---

## 🖌️ 이미지 필터 활용 예시

아래는 수직 엣지 감지, 수평 엣지 감지, 블러 처리, 샤프닝 필터 적용 예시입니다.

### 2. 수직 엣지 감지 (Vertical Edge Detection)
```python
# Sobel X를 이용한 수직 엣지 감지
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
plt.imshow(sobel_x, cmap='gray')
plt.title('Vertical Edge (Sobel X)')
plt.axis('off')
plt.show()
```

### 3. 수평 엣지 감지 (Horizontal Edge Detection)
```python
# Sobel Y를 이용한 수평 엣지 감지
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Horizontal Edge (Sobel Y)')
plt.axis('off')
plt.show()
```

### 4. 블러 처리 (Blurring)
```python
# Gaussian Blur를 이용한 블러 처리
blur = cv2.GaussianBlur(img, (5, 5), 0)
plt.imshow(blur, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')
plt.show()
```

### 5. 샤프닝 (Sharpening)
```python
# 샤프닝 필터 커널 정의 및 적용
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, kernel)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
plt.show()
```
