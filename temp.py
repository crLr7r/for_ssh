import numpy as np
from scipy.linalg import null_space

Del, lam, t_SO, t = 0, 0.2, 0.1, 1

Da = np.array([[Del, -1j*lam], [1j*lam, Del]])
Db = np.array([[-Del, -1j*lam], [1j*lam, -Del]])
T = np.array([[1j*t_SO, 0], [0, -1j*t_SO]])
I2 = np.eye(2)

I_half_a, I_half_b = np.eye(10), np.eye(10)
for i in range(2):
    I_half_a[2*i + 4, 2*i + 4] = 0
    I_half_b[2*i + 3, 2*i + 3] = 0
for i in range(3):
    I_half_a[i, i] = 0
    I_half_b[-(i+1), -(i+1)] = 0

SOC = np.array([[0,1,0,0,0,0,0,0,0,0],[-1,0,1,0,0,0,0,0,0,0],[0,-1,0,0,1,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0], [0,0,-1,0,0,0,-1,0,0,0],[0,0,0,1,0,0,0,1,0,0],
                [0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,-1,0,0,1,0],[0,0,0,0,0,0,0,-1,0,1],[0,0,0,0,0,0,0,0,-1,0]])
#Hop = np.array([[0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0]])
Hop = np.zeros((10,10), dtype=complex)
Hop[2][3], Hop[7][6] = -t, -t
for i in range(4):
    Hop[i + 3][i + 2], Hop[i + 3][i + 4] = -t, -t

A = np.kron(I_half_a,Da) + np.kron(I_half_b, Db) + np.kron(SOC, T) + np.kron(Hop, I2)
b = np.zeros((20,1))
trial_sol = np.array([[-0.0000000008-0.1192786488j, -0.0000000017+0.1192786488j,
                       0.0000000014+0.1572274701j, 0.0000000026-0.1572274701j,
                       0.0000000004-0.1783515692j, -0.0000000010+0.1783515692j,
                       0.0857555952+0.0000000003j, 0.0857555952+0.0000000004j,
                       -0.0000000011+0.1435021897j, -0.0000000000-0.1435021897j,
                       -0.1435021897+0.0000000000j, -0.1435021897-0.0000000011j,
                       0.0000000004-0.0857555952j, -0.0000000003+0.0857555952j,
                       0.1783515692+0.0000000010j, 0.1783515692+0.0000000004j,
                       -0.1572274701-0.0000000024j, -0.1572274701+0.0000000012j,
                       0.1192786488+0.0000000013j, 0.1192786488-0.0000000004j]])

trial_result = (A @ trial_sol.T)
wanted_result = (0.03544050999704 * trial_sol.T)

for i in range(len(trial_result)//2):
    up = trial_result[2 * i][0]
    down = trial_result[2 * i + 1][0]
    print(f"({up:.5f}, {down:.5f})")
#print("\n", wanted_result)
'''
for i in range(len(trial_result)//2):
    density = np.abs(trial_result[2*i][0])**2 + np.abs(trial_result[2*i+1][0])**2
    print(f"{density:.4f}", end=", ")
print("\n\n\n")
'''
#print(A)
U, S, Vh = np.linalg.svd(A)

#print("가장 작은 특이값:", S[-1])
sol = np.asarray(Vh[-1].T)
#sol = np.asarray(np.linalg.solve(A, b))
for elt in sol:
    print(elt)
#print(len(sol))
psi_square = []
for e in sol:
    psi_square.append(np.abs(e)**2)

density = np.asarray(psi_square[0::2]) + np.asarray(psi_square[1::2])
#for i, d in enumerate(density): print(rf"$|e_{i}|^2$ = {d}")

#print(np.linalg.det(A))
#print(sol)
'''
for i in (A*sol).flat:
    print(f"{i:.4f}", end=", ")
print("/")
for i in (1.6e-9 * sol).flat:
    print(f"{i:.4f}", end=", ")
print("/")
is_correct = np.allclose(A @ sol, b)
print("검증 결과:", is_correct)
'''
