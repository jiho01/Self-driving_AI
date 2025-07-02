# NumPy 기초 가이드

이 문서는 Python의 **NumPy** 라이브러리 핵심 기능과 활용 예제를 정리한 Markdown 파일입니다. NumPy는 과학 계산과 데이터 분석에 필수적인 다차원 배열 객체(`ndarray`)와 다양한 수치 연산 기능을 제공합니다.

---

## 목차

1. [NumPy 소개](#1-numpy-소개)
2. [설치 및 임포트](#2-설치-및-임포트)
3. [버전 확인](#3-버전-확인)
4. [배열 생성](#4-배열-생성)
5. [배열 속성](#5-배열-속성)
6. [데이터 타입](#6-데이터-타입)
7. [인덱싱 & 슬라이싱](#7-인덱싱--슬라이싱)
8. [브로드캐스팅](#8-브로드캐스팅)
9. [유니버설 함수 (ufunc)](#9-유니버설-함수-ufunc)
10. [복사 vs 뷰 (Copy vs View)](#10-복사-vs-뷰-copy-vs-view)
11. [배열 형태 변형 (Reshape)](#11-배열-형태-변형-reshape)
12. [배열 결합 & 분할](#12-배열-결합--분할)
13. [배열 검색 & 정렬](#13-배열-검색--정렬)
14. [배열 필터링](#14-배열-필터링)
15. [난수 생성](#15-난수-생성)
16. [선형대수 연산](#16-선형대수-연산)
17. [파일 입출력](#17-파일-입출력)
18. [성능 최적화 팁](#18-성능-최적화-팁)
19. [참고 자료](#19-참고-자료)

---

## 1. NumPy 소개

* `NumPy`(Numerical Python)는 대규모 다차원 배열과 행렬 연산을 지원하며, C 언어로 구현된 내부 알고리즘으로 속도가 매우 빠릅니다.
* 과학 계산, 머신러닝, 데이터 분석에서 기본 빌딩 블록 역할을 합니다.

## 2. 설치 및 임포트

```bash
pip install numpy
# 또는
conda install numpy
```

```python
import numpy as np
```

## 3. 버전 확인

```python
import numpy as np
print(np.__version__)
```

## 4. 배열 생성

```python
import numpy as np

# 1차원 배열
arr1 = np.array([1, 2, 3, 4])

# 2차원 배열
arr2 = np.array([[1, 2], [3, 4]])

# 초기값 배열
zeros = np.zeros((2, 3))      # 2×3 영행렬
ones  = np.ones((3, 2), dtype=int)  # 3×2 모두 1인 정수형
range_arr = np.arange(0, 10, 2)     # 0부터 8까지 2씩 증가
lin = np.linspace(0, 1, 5)          # 0~1 구간 균등 5개
```

## 5. 배열 속성

* `ndim`: 차원 수
* `shape`: 각 축의 크기 튜플
* `size`: 전체 요소 수
* `dtype`: 요소 데이터 타입
* `itemsize`: 각 요소의 바이트 크기

```python
print(arr2.ndim)    # 2
print(arr2.shape)   # (2, 2)
print(arr2.size)    # 4
print(arr2.dtype)   # dtype('int64')
print(arr2.itemsize) # 8
```

## 6. 데이터 타입

* `dtype` 속성으로 확인 가능
* `astype()`으로 변경

```python
arr = np.array([1, 2, 3], dtype=np.int32)
print(arr.dtype)
arr_f = arr.astype(np.float64)
print(arr_f.dtype)
```

## 7. 인덱싱 & 슬라이싱

```python
arr = np.arange(10)
print(arr[2])       # 2
print(arr[2:7:2])   # [2,4,6]

M = np.arange(9).reshape(3,3)
print(M[1,2])       # 5
print(M[:,1])       # [1,4,7]
```

> 슬라이스는 \*\*뷰(View)\*\*이므로, 슬라이스 결과를 수정하면 원본이 변함.

## 8. 브로드캐스팅

* 서로 다른 형태의 배열 간 연산을 자동으로 확장

```python
M = np.array([[1,2,3],[4,5,6]])
v = np.array([10,20,30])
print(M + v)
# [[11,22,33]
#  [14,25,36]]
```

## 9. 유니버설 함수 (ufunc)

* 벡터화된 수학 함수: `np.add`, `np.subtract`, `np.sqrt`, `np.sin` 등

```python
a = np.array([1,4,9,16])
print(np.sqrt(a))    # [1.,2.,3.,4.]
print(np.exp(a))     # 지수 함수
```

## 10. 복사 vs 뷰 (Copy vs View)

```python
arr = np.arange(5)
view = arr[2:]
copy = arr.copy()
view[0] = 99        # arr[2]도 변경됨
copy[0] = 100       # arr는 변경되지 않음
```

## 11. 배열 형태 변형 (Reshape)

```python
arr = np.arange(6)
reshaped = arr.reshape((2,3))
flattened = reshaped.ravel()  # 1차원으로 펼침
transposed = reshaped.T      # 전치
```

## 12. 배열 결합 & 분할

```python
# 결합
a = np.array([1,2])
b = np.array([3,4])
print(np.concatenate([a,b]))    # [1,2,3,4]
print(np.vstack([a,b]))         # 세로 결합
print(np.hstack([a,b]))         # 가로 결합

# 분할
print(np.split(np.arange(6), [2,4]))
```

> `np.array_split`은 균등 분할 불가능 시에도 사용

## 13. 배열 검색 & 정렬

```python
arr = np.array([6,2,8,4,10])
print(np.where(arr>5))    # 조건 만족 인덱스 반환
print(np.sort(arr))       # [2,4,6,8,10]
```

## 14. 배열 필터링

* Boolean indexing

```python
arr = np.arange(10)
filtered = arr[arr%2==0]  # 짝수만
```

## 15. 난수 생성

```python
# 균등분포
print(np.random.rand(2,3))
# 정규분포
print(np.random.randn(2,3))
# 정수 난수
print(np.random.randint(0,10,size=(2,3)))
# 시드 고정
np.random.seed(42)
```

## 16. 선형대수 연산

```python
A = np.array([[1,2],[3,4]])
print(np.dot(A,A))       # 행렬 곱
print(np.linalg.inv(A))  # 역행렬
print(np.linalg.det(A))  # 행렬식
```

> 고급 연산: `np.eig`, `np.linalg.svd` 등

## 17. 파일 입출력

```python
# 텍스트 파일
np.savetxt('data.txt', arr, fmt='%d')
loaded = np.loadtxt('data.txt', dtype=int)

# 바이너리(.npy)
np.save('arr.npy', arr)
arr2 = np.load('arr.npy')
```

## 18. 성능 최적화 팁

* 가능한 **벡터화(vectorization)** 사용
* Python 루프 대신 NumPy 함수·연산 활용
* `numba` 등 JIT 컴파일러 병행 고려

## 19. 참고 자료

* [NumPy 공식 문서](https://numpy.org/doc/)
* [W3Schools NumPy Tutorial](https://www.w3schools.com/python/numpy/default.asp)
* [SciPy Lectures: NumPy](https://scipy-lectures.org/)
