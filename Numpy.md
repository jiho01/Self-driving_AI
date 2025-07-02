# NumPy 기초 가이드

이 문서는 Python에서 과학 계산을 빠르고 효율적으로 수행할 수 있게 해주는 `NumPy` 라이브러리의 기초 내용을 정리한 Markdown 파일입니다.

---

## 1. NumPy란?

* `NumPy`(Numerical Python)는 다차원 배열 객체(`ndarray`)와 다양한 수치 연산 함수를 제공하는 라이브러리입니다.
* Python 리스트에 비해 배열 연산 속도가 빠르고 메모리 효율이 뛰어납니다.

## 2. 설치 방법

```bash
# pip 사용
pip install numpy

# conda 사용
conda install numpy
```

## 3. 배열(Array) 생성

```python
import numpy as np

# 1차원 배열 생성
arr1 = np.array([1, 2, 3, 4])

# 2차원 배열 생성
arr2 = np.array([[1, 2], [3, 4]])

# 초기값을 사용한 배열 생성
zeros = np.zeros((2, 3))      # 2x3 영행렬
ones = np.ones((3, 2), dtype=int)  # 3x2에 값 1, 정수형
range_arr = np.arange(0, 10, 2)     # 0부터 8까지 2씩 증가
lin = np.linspace(0, 1, 5)          # 0~1 범위에서 5개 균등 분할
```

## 4. 배열 속성

* `ndim`: 배열 차원 수
* `shape`: 각 차원의 크기 튜플
* `size`: 전체 요소 수
* `dtype`: 요소 데이터 타입
* `itemsize`: 각 요소의 바이트 크기

```python
print(arr2.ndim)    # 2
print(arr2.shape)   # (2, 2)
print(arr2.dtype)   # dtype('int64')
print(arr2.size)    # 4
```

## 5. 기본 연산 및 브로드캐스팅

* 배열 간 산술 연산은 요소별(element-wise)로 수행됩니다.
* 서로 다른 크기의 배열도 브로드캐스팅 규칙에 따라 연산이 가능합니다.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)   # [5 7 9]
print(a * 2)   # [2 4 6]

# 브로드캐스팅 예시
M = np.array([[1, 2, 3],
              [4, 5, 6]])
v = np.array([10, 20, 30])
print(M + v)
# [[11 22 33]
#  [14 25 36]]
```

## 6. 인덱싱(Indexing)과 슬라이싱(Slicing)

```python
arr = np.arange(10)
print(arr[2])       # 2
print(arr[2:7:2])   # [2 4 6]

M = np.arange(9).reshape(3, 3)
print(M[1, 2])      # 5
print(M[:, 1])      # [1 4 7] (두 번째 열)
```

## 7. 배열 형태 변형

* `reshape()`: 새로운 형태로 재배열
* `flatten()` 또는 `ravel()`: 1차원으로 펼치기
* `transpose()`: 전치
* `concatenate()`, `vstack()`, `hstack()`: 배열 결합

```python
M = np.arange(6)
M2 = M.reshape((2, 3))
flat = M2.flatten()
T = M2.T

# 행 결합, 열 결합
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.vstack([a, b])  # 세로 결합
h = np.hstack([a, b])  # 가로 결합
```

## 8. 통계 및 수학 함수

```python
data = np.random.randn(1000)
print(data.mean())   # 평균
print(data.std())    # 표준편차
print(data.sum())    # 합계
print(data.min(), data.max())
print(np.percentile(data, [25, 50, 75]))  # 분위수
```

## 9. 난수 생성 (Random)

```python
# 균등분포
u = np.random.rand(3, 2)

# 정규분포
n = np.random.randn(3, 2)

# 정수 난수
r = np.random.randint(low=0, high=10, size=(3,3))

# 시드 설정 (재현성)
np.random.seed(42)
```

## 10. 유용한 함수 요약

| 함수              | 설명            |
| --------------- | ------------- |
| `np.unique`     | 배열의 고유값 반환    |
| `np.where`      | 조건에 맞는 인덱스 반환 |
| `np.sort`       | 배열 정렬         |
| `np.dot`, `@`   | 행렬 곱          |
| `np.linalg.inv` | 행렬 역행렬        |
| `np.linalg.det` | 행렬 행렬식        |

---

## 참고 자료

* 공식 문서: [https://numpy.org/doc/](https://numpy.org/doc/)
* NumPy 튜토리얼: [https://numpy.org/devdocs/user/quickstart.html](https://numpy.org/devdocs/user/quickstart.html)

