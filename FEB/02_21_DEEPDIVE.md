# 4주차 딥다이브 - Pytorch 벡터 연산과 텐서 연산의 차이를 설명하시오

### 💡Pytorch 에서의 `백터 연산과 텐서 연산의 차이`를 살펴보기 전에 `벡터와 텐서라는 개념의 차이`를 확인하자

</aside>

# 벡터 vs 텐서

`Pytorch`에서의 텐서와 벡터는 개념적으로 구분되어 있다는 느낌을 받지 못했지만, 벡터 형태인 경우와 텐서(다차원 배열)의 형태에 따라 연산의 차이가 있다.

→ 1D tensor는 벡터

→ 2D tensor는 행렬

→ 그 이상의 n-D tensor를 텐서  

## 벡터

정의: 벡터(vector)란 크기와 방향을 가지고 있는 양을 나타내는 개념이다. 벡터는 크기와 방향이 중요한 개념

표현: 수학적으로 벡터는 열(column)의 형태로 나타낼 수 있다. 크기가 2인 벡터는 하기와 같음

$$
v = [v_1 \,\,v_2]
$$

```python
# 파이토치에서는 tensor로 벡터를 생성 가능
# type은 텐서이지만 벡터의 형태를 보여줌
import torch

a_vec = torch.tensor([1,3], dtype = torch.float32)
print(a_vec.type())

>> torch.FloatTensor # 출력
```

`torch.tensor`는 생성 시 데이터 타입을 지정해주지 않으면 LongTensor로 구현됨

`LongTensor` 는 정수 타입이다.

⇒ `Pytorch` 텐서에서 활용된는 데이터 타입을 공식 도큐먼트에서 확인 가능

### (참고) 딥러닝 모델에 입력을 float 텐서로 넣어주는 것이 좋다?

1. 딥러닝 모델은 대부분 부동소수점 연산(Floating Point Arithmetic)을 기반으로 동작
2. `torch.float32` 또는 `torch.float16`을 사용하면 GPU/CPU에서 최적화된 연산을 수행
3. `Pytorch`의 자동 미분(Autograd)은 float텐서에서 동작함
    1. `int` 또는 `bool` 텐서는 미분 계산이 불가능하며, backward(Backpropagation) 수행 시 오류 발생 https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html

https://pytorch.org/docs/stable/tensors.html

---

### 텐서(Tensor)

정의: 데이터를 담기 위한 컨테이너로 다차원 배열의 형태이다. 일반적으로 수치형 데이터를 저장하고, 동적 크기를 같는다. 

- 1D tensor는 벡터
- 2D tensor는 행렬
- 그 이상의 n-D tensor를 텐서
    - 3D 텐서는 일반적으로 큐브 형태와 같은 모양으로 세개의 축이 존재하며 주로 샘픔과 특징(feature)를 갖는 구조
    - 4D 텐서는 컬러 이미지가 대표적인 사례로 흑백 이미지의 경우 3D 텐서로 표현이 가능하다. 채널(channel)을 갖는 구조로 사용
    - 5D 텐서는 비디오 데이터가 대표적이다.
    - ND 텐서…

---

## 텐서와 벡터 연산의 차이 in Pytorch

`Pytorch`에서 벡터 연산과 텐서 연산이 수행되는 메커니즘에는 몇 가지 중요한 차이가 있다. 기본적으로 벡터 연산은 1차원 텐서(즉, shape이 `(N,)`인 텐서)에서 이루어지지만, 텐서 연산은 더 일반적으로 다차원 배열에서 수행됨

### 1. **연산 단위의 차이**

- **벡터 연산**: 1D 텐서(`torch.tensor([a, b, c])`)에서 수행되며, 일반적으로 선형 대수 연산(예: 내적, 외적, L2 Norm 등)에 해당
- **텐서 연산**: 다차원(2D, 3D, N-D) 텐서에서 수행되며, 행렬 연산, 브로드캐스팅, 축을 따라 수행되는 연산 등이 포함

### 2. **연산의 내부 최적화 차이**

- `Pytorch`는 연산을 수행할 때 **벡터 연산을 SIMD (Single Instruction, Multiple Data) 방식으로 병렬화하며,** 이는 CPU 또는 GPU에서 수행되는 연산이 **벡터화**되어 더 빠르게 실행될 수 있음을 의미
- 텐서 연산에서는 **CUDA 커널, 병렬 연산, 브로드캐스팅** 등이 적용되며, 연산 방식이 더 복잡하다. 예를 들어, 행렬 곱(`torch.matmul`)은 GPU에서 내부적으로 **최적화된 블록 연산과 공유 메모리 활용 기법**을 사용

### 3. **브로드캐스팅(Broadcasting)의 차이**

- **벡터 연산에서는 보통 브로드캐스팅이 필요하지 않음**: 두 벡터 간 연산은 같은 크기일 때 수행됨
- **텐서 연산에서는 브로드캐스팅이 필수적일 수 있음**: 서로 다른 차원의 텐서가 연산될 경우 Pytorch는 자동으로 브로드캐스팅을 수행하여 차원을 맞춘 후 연산을 수행

```python
a = torch.tensor([1, 2, 3])  # (3,)
b = torch.tensor([[1], [2], [3]])  # (3,1)
c = a + b  # 브로드캐스팅 발생 (3,) -> (3,1)으로 확장되어 연산됨
print(c.shape)  # 결과: (3,3)

```

<aside>
💡

브로드캐스팅의 경우, 에러가 뜨지 않고 계산이 수행된다는 장점이 있지만 본인이 의도한 계산인지는 검증이 **필수적**으로 필요하다.

</aside>

### 4. **메모리 할당 및 연산 방식 차이**

- 벡터 연산의 경우 일반적으로 연산이 작은 크기의 메모리 블록에서 수행되며, 연산 비용이 낮음
- 텐서 연산은 **큰 차원의 데이터에 대해 다양한 차원 축을 따라 연산이 수행되기 때문에 메모리 레이아웃(Layout), 캐시 활용, 데이터 접근 방식이 최적화됨**

### 5. **GPU 연산 차이**

- 벡터 연산은 GPU에서 단순한 **CUDA 벡터화 연산 (warp 기반 연산)** 으로 실행
- 텐서 연산은 GPU에서 **텐서 코어(Tensor Cores)** 를 활용하는 고성능 연산 방식(예: `torch.matmul`)을 사용하며, 연산 성능이 크게 향상될 수 있음

### 6. **특정 연산의 내부 처리 차이**

- 예를 들어 `torch.dot()`은 벡터 내적 연산만 지원하지만, `torch.matmul()` 또는 `@` 연산자는 **벡터·행렬·고차원 텐서 연산**을 모두 지원 → 실제 Transformer의 self-attention 계산 시에도 `torch.matmul()`이 아닌 `@` 연산자를 사용함, `Tensorflow`에서는 `tf.matmul()` 활용

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("벡터 내적 결과:",torch.dot(a, b))  # 벡터 내적

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print("torch.matmul() 결과:",torch.matmul(A, B))  # 행렬 곱셈

print("행렬곱 수행:\n", A @ B)

>> # 출력 결과

벡터 내적 결과:
 tensor(32)
torch.matmul() 결과:
 tensor([[19, 22],
        [43, 50]])
행렬곱 수행(@):
 tensor([[19, 22],
        [43, 50]])

```
