import numpy as np
from scipy.linalg import null_space

Del, lam, t_SO, t = 0, 0.33, 0.1, 1

Da = np.array([[Del, -1j*lam], [1j*lam, Del]])
Db = np.array([[-Del, -1j*lam], [1j*lam, -Del]])
T = np.array([[1j*t_SO, 0], [0, -1j*t_SO]])
I2 = np.eye(2)

I_half_a, I_half_b = np.eye(10), np.eye(10)
I_half_a[4][4], I_half_a[6][6] = 0, 0
I_half_b[3][3], I_half_b[5][5] = 0, 0
for i in range(3):
    I_half_a[i, i] = 0
    I_half_b[-(i+1), -(i+1)] = 0

SOC = np.array([[0,1,0,0,0,0,0,0,0,0],[-1,0,1,0,0,0,0,0,0,0],[0,-1,0,0,1,0,0,0,0,0],[0,0,0,0,0,-1,0,0,0,0], [0,0,-1,0,0,0,-1,0,0,0], 
                [0,0,0,1,0,0,0,1,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,-1,0,0,1,0],[0,0,0,0,0,0,0,-1,0,1],[0,0,0,0,0,0,0,0,-1,0]])
#Hop = np.array([[0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0]])
Hop = np.zeros((10,10), dtype=complex)
Hop[2][3], Hop[7][6] = -t, -t
for i in range(4):
    Hop[i+3][i+2], Hop[i+3][i+4] = -t, -t

A = np.kron(I_half_a,Da) + np.kron(I_half_b, Db) + np.kron(SOC, T) + np.kron(Hop, I2)
b = np.zeros((12,1))
trial_sol = np.array([[0.1039-0.0845j, 0.0845+0.1039j, 0.0347+0.1494j, 0.0645-0.1391j,0.0307-0.1985j, -0.0105+0.2006j,0.0965+0.0024j, 0.0950+0.0174j,-0.0388+0.1855j, -0.0000-0.1895j,-0.1895+0.0000j, -0.1855-0.0388j,0.0174-0.0950j, -0.0024+0.0965j,0.2006+0.0105j, 0.1985+0.0307j,-0.1391-0.0645j, -0.1494+0.0347j,0.1039-0.0845j, 0.0845+0.1039j]])

trial_result = (A @ trial_sol.T)
print(len(trial_sol[0]))
print(len(trial_result))
for i in range(len(trial_result)//2):
    density = np.abs(trial_result[2*i][0])**2 + np.abs(trial_result[2*i+1][0])**2
    print(f"{density:.4f}", end=", ")
print("\n\n\n")
print(A)
U, S, Vh = np.linalg.svd(A)

#print("가장 작은 특이값:", S[-1])
sol = np.asarray(Vh[-1].T)
#sol = np.asarray(np.linalg.solve(A, b))

#print(len(sol))
psi_square = []
for e in sol:
    psi_square.append(np.abs(e)**2)

density = np.asarray(psi_square[0::2]) + np.asarray(psi_square[1::2])
for i, d in enumerate(density): print(rf"$|e_{i}|^2$ = {d}")

#print(np.linalg.det(A))
#print(sol)

for i in (A*sol).flat:
    print(f"{i:.4f}", end=", ")
print("/")
for i in (1.6e-9 * sol).flat:
    print(f"{i:.4f}", end=", ")
print("/")
is_correct = np.allclose(A @ sol, b)
print("검증 결과:", is_correct)

