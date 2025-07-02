1. Introduction
NumPy란?:

“Numerical Python”의 줄임말로, 고성능 다차원 배열 객체(ndarray)와 수치 연산 함수들을 제공

Python 리스트 대비 연산 속도·메모리 효율이 월등히 우수 
geeksforgeeks.org

2. Getting Started
설치 및 임포트:

bash
코드 복사
pip install numpy
python
코드 복사
import numpy as np
버전 확인:

python
코드 복사
np.__version__
이 한 줄만으로도 NumPy가 잘 설치됐는지, 어떤 버전을 쓰고 있는지 바로 확인 가능 
geeksforgeeks.org

3. Creating Arrays
기본 생성:

python
코드 복사
arr = np.array([1, 2, 3, 4])
초기값 배열:

np.zeros((행, 열)), np.ones((행, 열), dtype=int)

np.arange(start, stop, step), np.linspace(start, stop, num)

예시:

python
코드 복사
zeros = np.zeros((2,3))      # 2×3 영행렬  
ones  = np.ones((3,2), int)  # 3×2 모두 1인 정수형  
``` :contentReference[oaicite:3]{index=3}  
4. Array Indexing
단일 요소 접근:

python
코드 복사
x = arr[2]       # 3번째 요소  
다차원 배열:

python
코드 복사
M = np.array([[1,2,3],[4,5,6]])
M[1,2]           # 2행 3열 요소  
``` :contentReference[oaicite:4]{index=4}  
5. Array Slicing
슬라이스 문법:

python
코드 복사
arr[2:7:2]       # 인덱스 2부터 6까지 2칸씩 건너뛰기  
M[:,1]           # 모든 행의 2열  
뷰(View): 슬라이스 결과는 원본 배열의 뷰(view)로, 값을 바꾸면 원본에도 반영 
w3schools.com

6. Data Types
dtype 속성: 배열 요소의 자료형 확인

python
코드 복사
arr.dtype
형 변환:

python
코드 복사
arr.astype(float)
``` :contentReference[oaicite:6]{index=6}  
7. Copy vs View
얕은 복사(View): 원본과 메모리 공유

깊은 복사(Copy): arr.copy()로 완전 복제 (원본 변경 무관) 
w3schools.com

8. Array Shape & Reshape
shape:

python
코드 복사
arr.shape        # (행, 열, …) 튜플  
reshape():

python
코드 복사
arr.reshape((2,3))
차원 변경 시 요소 수 일치 필요 
w3schools.com

9. Array Iterating
for문 사용:

python
코드 복사
for x in arr:
    print(x)
다차원 반복: 중첩 for문 또는 np.nditer() 활용 
w3schools.com

10. Array Join & Split
결합:

np.concatenate([a,b]), np.vstack(), np.hstack()

분할:

np.split(), np.array_split() 
w3schools.com

11. Array Search, Sort & Filter
검색: np.where(condition)로 조건에 맞는 인덱스 반환

정렬: np.sort()

필터링: Boolean indexing

python
코드 복사
arr[arr > 5]
``` :contentReference[oaicite:11]{index=11}  
12. Random 모듈 / ufunc 등
Random: np.random.rand(), randint(), seed() 등

ufunc: 벡터화된 수학 함수 (np.add, np.sqrt, np.sin…)

더 자세한 내용은 각 섹션 참고 
w3schools.com
