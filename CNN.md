from pathlib import Path

# Markdown content as defined earlier
markdown_content = """
# 🧠 Convolutional Neural Network (CNN)

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
2. ReLU (Rectified Linear Unit)
비선형 활성화 함수로, 모델에 비선형성을 부여합니다.

음수는 0으로, 양수는 그대로 유지합니다.

python
항상 세부 정보 표시

복사
# 예시
F.relu(x)
3. Pooling Layer (풀링층)
특징 맵의 크기를 줄여 연산량을 줄이고, 특징의 위치 변화에 대해 불변성을 제공합니다.

종류: Max Pooling, Average Pooling 등

python
항상 세부 정보 표시

복사
# 예시 (Max Pooling)
nn.MaxPool2d(kernel_size=2, stride=2)
4. Fully Connected Layer (완전 연결층)
마지막 단계에서 추출된 특징을 기반으로 분류(Classification) 를 수행합니다.

일반적인 인공신경망(Dense Layer)와 동일한 구조입니다.

python
항상 세부 정보 표시

복사
# 예시
nn.Linear(in_features=128, out_features=10)
🔄 전체 흐름 예시
scss
항상 세부 정보 표시

복사
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
🎯 CNN의 특징
이미지의 2차원 구조를 그대로 유지하며 처리

파라미터 수가 적고 계산 효율이 높음

특징을 자동으로 추출함 (수작업 특징 설계 필요 없음)

영상 처리, 의료 영상, 자율주행, 얼굴 인식 등에서 폭넓게 활용됨

📚 주요 활용 분야
분야	예시
이미지 분류	고양이 vs 강아지, 숫자 인식
객체 탐지	자율주행 자동차, CCTV 분석
얼굴 인식	Face ID, 감정 분석
의료 영상	CT/MRI 이상 탐지

🧪 실습 예시
python
항상 세부 정보 표시

복사
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)  # MNIST용

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x
