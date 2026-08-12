import numpy as np
from scipy.linalg import null_space

Del, lam, t_SO, t = 0, 0.2, 0.1, 1

D = np.array([[Del, -1j*lam], [1j*lam, Del]])
D2 = np.array([[-Del, -1j*lam], [1j*lam, -Del]])
T = np.array([[1j*t_SO, 0], [0, -1j*t_SO]])
I2 = np.eye(2)

I_half1, I_half2 = np.eye(8), np.eye(8)
for i in range(4):
    I_half2[i, i] = 0
    I_half1[i+4, i+4] = 0

SOC = np.array([[0,1,0,0,0,0,0,0], [-1,0,1,0,0,0,0,0],[0,-1,0,1,0,0,0,0],[0,0,-1,0,0,0,0,0], [0,0,0,0,0,-1,0,0], [0,0,0,0,1,0,-1,0],[0,0,0,0,0,1,0,-1],[0,0,0,0,0,0,1,0]])
#Hop = np.array([[0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0]])
Hop = np.array([[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])

A = np.kron(I_half1,D) + np.kron(I_half2, D2) + np.kron(SOC, T) + np.kron(Hop, -t*I2)
b = np.zeros((16,1))

# A 행렬에서 1번 행과 1번 열을 제외한 부분행렬 A_prime
A_prime = A[1:, 1:]
b_prime = -A[1:, 0]  # x1 = 1 로 둔 결과

# 이제 det(A_prime) != 0 이라면 np.linalg.solve 로 0이 아닌 해를 얻을 수 있음
#x_sub = np.linalg.solve(A_prime, b_prime)
#sol = np.insert(x_sub, 0, 1.0) # x1 = 1 결합
#sol = np.linalg.solve(A, b)
U, S, Vh = np.linalg.svd(A)

# S에는 특이값(Singular value)들이 내림차순으로 들어있음
# 마지막 특이값(S[-1])이 0에 얼마나 가까운지 확인해보세요!
print("가장 작은 특이값:", S[-1])

# 가장 작은 특이값에 대응하는 오른쪽 특이벡터가 바로 '최적의 Non-trivial 해'입니다.
sol = Vh[-1].T

a1_a0 = np.abs(sol[3] + sol[2])**2 - np.abs(sol[1] + sol[0])**2

print(np.linalg.det(A))
print(sol)
print(a1_a0)

# A @ x 연산 결과가 b와 일치하는지 검증
is_correct = np.allclose(A @ sol, b)
print("검증 결과:", is_correct)  # True
#print(a1_a0)
