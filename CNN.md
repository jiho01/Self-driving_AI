# 🧠 Convolutional Neural Network (CNN) 및 이미지 처리·표시 방법

CNN(Convolutional Neural Network)은 이미지나 영상 같은 시각 데이터를 처리하기 위해 설계된 딥러닝 모델입니다. 주로 **이미지 분류, 객체 인식, 자율주행, 얼굴 인식** 등에 사용됩니다.

---

## 📝 주요 용어 정리

| 용어                      | 설명                                                    |
| ----------------------- | ----------------------------------------------------- |
| **Convolution**         | 필터(kernel)를 입력 데이터에 슬라이딩하며 국부 영역의 가중 합을 계산하는 연산       |
| **Kernel / Filter**     | 입력 이미지에서 특징을 추출하기 위한 작은 행렬(예: 3×3)                    |
| **Feature Map**         | 커널을 적용해 얻은 출력 행렬. 이미지의 특정 특징이 강조된 형태                  |
| **Stride**              | 커널을 적용할 때 이동하는 보폭(간격)                                 |
| **Padding**             | 입력 경계에서 커널 적용을 위해 입력 주변에 0 등을 추가하는 것                  |
| **Activation Function** | 뉴런의 출력에 비선형성을 추가하는 함수(예: ReLU, Sigmoid)               |
| **Pooling**             | 특징 맵 크기를 줄여 연산량을 감소시키고 위치 변화에 강인성을 제공(예: Max Pooling) |
| **Flatten**             | 다차원 배열을 1차원 벡터로 변환하는 과정                               |
| **Epoch**               | 전체 훈련 데이터셋을 한 번 학습하는 주기                               |
| **Batch Size**          | 한 번에 모델에 입력되어 학습되는 샘플 수                               |
| **Learning Rate**       | 가중치 업데이트 시 이동하는 크기를 조절하는 하이퍼파라미터                      |

---

## 🔧 주요 구성 요소 및 용어 정리

### 1. Convolution Layer (합성곱 층)

* 위의 **Convolution**, **Kernel**, **Feature Map**, **Stride**, **Padding** 용어를 사용하여 이미지 특징을 추출합니다.
* 예시 (PyTorch):

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

### 2. ReLU (Rectified Linear Unit)

* 음수는 0으로, 양수는 그대로 통과시키는 비선형 활성화 함수.

```python
F.relu(x)
```

### 3. Pooling Layer (풀링층)

* Max Pooling으로 특징 맵 다운샘플링:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

### 4. Fully Connected Layer (완전 연결층)

* Flatten된 벡터를 입력으로 받아 최종 분류를 수행합니다.

```python
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

* 이미지의 **2차원 구조를 그대로 유지**하며 처리
* 파라미터 수가 적고 계산 효율이 높음
* 특징을 **자동으로 추출**함 (수작업 특징 설계 필요 없음)
* 영상 처리, 의료 영상, 자율주행, 얼굴 인식 등에서 폭넓게 활용됨

---

## 📚 주요 활용 분야

| 분야     | 예시                |
| ------ | ----------------- |
| 이미지 분류 | 고양이 vs 강아지, 숫자 인식 |
| 객체 탐지  | 자율주행 자동차, CCTV 분석 |
| 얼굴 인식  | Face ID, 감정 분석    |
| 의료 영상  | CT/MRI 이상 탐지      |

---

## 🖼️ 이미지 표시 방법

### 1. Matplotlib 사용하기

```python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('내이미지.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

### 2. OpenCV 자체 윈도우 (로컬 스크립트)

```python
import cv2

img = cv2.imread('내이미지.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. Colab / Jupyter 환경에서 cv2\_imshow 사용

```python
from google.colab.patches import cv2_imshow
import cv2

img = cv2.imread('내이미지.jpg')
cv2_imshow(img)
```

---

## 🖌️ 이미지 필터 활용 예시

### 수직 엣지 감지 (Vertical)

```python
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')
plt.show()
```

### 수평 엣지 감지 (Horizontal)

```python
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
plt.imshow(sobel_y, cmap='gray')
plt.axis('off')
plt.show()
```

### 블러 처리 (Blurring)

```python
blur = cv2.GaussianBlur(img, (5, 5), 0)
plt.imshow(blur, cmap='gray')
plt.axis('off')
plt.show()
```

### 샤프닝 (Sharpening)

```python
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, kernel)
plt.imshow(sharpened, cmap='gray')
plt.axis('off')
plt.show()
```
