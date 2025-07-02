# NumPy 기초 가이드

Python에서 과학 계산과 데이터 분석을 위해 널리 쓰이는 **NumPy** 라이브러리의 핵심 개념과 활용 예제를 정리했습니다. 각 섹션마다 개념을 매끄럽게 설명한 뒤, 바로 연습할 수 있는 코드 예시를 제공합니다.

---

## 목차

1. [NumPy의 철학과 장점](#1-numpys-철학과-장점)
2. [`ndarray`의 구조와 메모리](#2-ndarray의-구조와-메모리)
3. [브로드캐스팅 원리](#3-브로드캐스팅-원리)
4. [벡터화와 유니버설 함수](#4-벡터화와-유니버설-함수)
5. [배열 생성과 초기화](#5-배열-생성과-초기화)
6. [인덱싱(Indexing)](#6-인덱싱indexing)
7. [슬라이싱(Slicing)](#7-슬라이싱slicing)
8. [팬시 인덱싱(Fancy Indexing)](#8-팬시-인덱싱fancy-indexing)
9. [배열 형태 변형(Reshape)](#9-배열-형태-변형reshape)
10. [결합 및 분할(Concatenate & Split)](#10-결합-및-분할concatenate--split)
11. [통계 및 집계 함수](#11-통계-및-집계-함수)
12. [난수 생성(Random)](#12-난수-생성random)
13. [선형대수 연산(Linear Algebra)](#13-선형대수-연산linear-algebra)
14. [파일 입출력(I/O)](#14-파일-입출력io)
15. [성능 최적화 팁](#15-성능-최적화-팁)
16. [참고 자료](#16-참고-자료)

---

## 1. NumPy의 철학과 장점

NumPy는 다차원 배열(`ndarray`) 연산을 C로 최적화하여 파이썬 루프 대비 수십 배 빠른 성능을 제공합니다. 동일 자료형을 연속 메모리에 저장해 메모리 오버헤드를 줄이고, Pandas나 SciPy 등 데이터 과학 생태계의 기본 토대를 이룹니다.

```bash
# 설치 (pip 또는 conda)
pip install numpy
# 파이썬에서 불러오기
import numpy as np
```

## 2. `ndarray`의 구조와 메모리

`ndarray`는 아래 속성으로 배열을 설명합니다:

* **shape**: 각 축(axis)의 크기
* **dtype**: 요소 자료형
* **strides**: 차원별 메모리 건너뛰기(byte 단위)

스트라이드는 배열이 메모리에 어떻게 배치되는지 알려주며, 연속 메모리 배열(C-contiguous)인 경우 연산 최적화에 유리합니다.

```python
import numpy as np
arr = np.arange(8).reshape(2, 4)
print(arr.shape)   # (2, 4)
print(arr.dtype)   # int64
print(arr.strides) # (32, 8)
```

## 3. 브로드캐스팅 원리

서로 다른 형태(shape)의 배열끼리도 연산을 가능하게 하는 규칙입니다. 뒤쪽 차원부터 비교해 크기가 같거나 1인 축을 자동 확장합니다.

```python
M = np.ones((3, 4))
v = np.array([1, 2, 3, 4])  # shape (4,)
print(M + v)  # v가 각 행에 더해짐
```

## 4. 벡터화와 유니버설 함수

벡터화(vectorization)는 파이썬 레벨의 반복문 없이 C 레벨에서 일괄 처리해 빠른 성능을 내며, ufunc는 요소별 수학 연산을 제공합니다.

```python
arr = np.arange(5)
print(np.sin(arr))      # 사인 함수
print(np.add(arr, 10))  # 모든 요소에 10 더하기
```

## 5. 배열 생성과 초기화

다양한 방식으로 배열을 만들 수 있습니다:

* `np.array` : 리스트 → ndarray
* `np.zeros`, `np.ones` : 0 또는 1로 채운 배열
* `np.empty` : 초기화 없이 메모리 할당
* `np.arange`, `np.linspace` : 규칙적 수열
* `np.eye`, `np.full` : 단위행렬, 특정 값으로 채움

```python
zeros = np.zeros((2,3))
ones  = np.ones((3,2), int)
ev_2s = np.arange(0, 10, 2)
points = np.linspace(0, 1, 5)
```

## 6. 인덱싱(Indexing)

인덱싱은 배열의 특정 위치에서 단일 요소를 선택합니다. 1차원과 다차원 모두 대괄호로 접근합니다.

```python
arr = np.array([10, 20, 30, 40])
print(arr[2])  # 30

M = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(M[1,2])  # 6
```

## 7. 슬라이싱(Slicing)

슬라이싱은 연속된 구간을 잘라내 뷰(view)를 생성합니다. 뷰를 수정하면 원본도 바뀝니다.

```python
arr = np.arange(10)
sub = arr[2:8:2]  # [2, 4, 6]

M = np.arange(9).reshape(3,3)
col2 = M[:,1]     # 두 번째 열 [1, 4, 7]
```

## 8. 팬시 인덱싱(Fancy Indexing)

정수 배열이나 불리언 배열로 여러 위치를 선택해 복사본을 만듭니다. 원본은 변경되지 않습니다.

```python
arr = np.arange(10)
indices = [1,3,5]
selected = arr[indices]   # [1, 3, 5]
mask = arr % 2 == 0
evens = arr[mask]         # [0, 2, 4, 6, 8]
```

## 9. 배열 형태 변형(Reshape)

`reshape`으로 차원을 변경하고, `ravel`은 뷰, `flatten`은 복사본으로 1차원 배열을 만듭니다.

```python
x = np.arange(12)
mat = x.reshape(3,4)
flat_view = mat.ravel()
flat_copy = mat.flatten()
```

## 10. 결합 및 분할(Concatenate & Split)

`concatenate`, `vstack`, `hstack`으로 합치고, `split`, `array_split`으로 분할합니다.

```python
A = np.array([[1,2]])
B = np.array([[3,4]])
C = np.vstack([A, B])
parts = np.split(np.arange(6), [2,5])
```

## 11. 통계 및 집계 함수

평균, 합계, 최소/최대, 분위수 등을 한 줄로 계산합니다.

```python
data = np.random.randn(1000)
print(data.mean(), data.std())
print(np.percentile(data, [25,50,75]))
```

## 12. 난수 생성(Random)

`default_rng` API를 이용해 균등분포, 정규분포, 정수 난수를 생성하고, 재현 가능한 시드를 설정합니다.

```python
from numpy.random import default_rng
gen = default_rng(42)
print(gen.random((2,3)))
print(gen.integers(0,10, size=5))
```

## 13. 선형대수 연산(Linear Algebra)

`numpy.linalg`에서 행렬 곱, 역행렬, 고유값 분해, SVD 등을 지원합니다.

```python
from numpy.linalg import inv, det, eig, svd
A = np.array([[4,0],[0,4]])
print(inv(A), det(A))
w, v = eig(A)
U, s, Vt = svd(A)
```

## 14. 파일 입출력(I/O)

`loadtxt`/`savetxt`로 텍스트, `save`/`load`, `savez`/`load`로 바이너리 파일을 다룹니다.

```python
np.savetxt('data.txt', mat, fmt='%d')
mat2 = np.loadtxt('data.txt', dtype=int)
np.save('mat.npy', mat)
archives = np.savez('all.npz', a=mat, b=C)
loaded = np.load('all.npz')
```

## 15. 성능 최적화 팁

* **벡터화**로 파이썬 루프 최소화
* **연속 메모리** 배열(C-contiguous) 활용
* 적절한 **dtype** 선택으로 메모리 절약
* 멀티스레드 BLAS(LAPACK) 라이브러리 이용

