import numpy as np
from scipy.linalg import null_space

Del, lam, t_SO, t = 0, 0.2, 0.1, 1

Da = np.array([[Del, -1j*lam], [1j*lam, Del]])
Db = np.array([[-Del, -1j*lam], [1j*lam, -Del]])
T = np.array([[1j*t_SO, 0], [0, -1j*t_SO]])
I2 = np.eye(2)

I_half_a, I_half_b = np.eye(12), np.zeros((12, 12))

for i in range(3):
    I_half_a[i, i] = 0
    I_half_b[i, i] = 1
    I_half_a[-(i+1), -(i+1)] = 0
    I_half_b[-(i+1), -(i+1)] = 1

SOC = np.array([[0,1,0,0,0,0,0,0,0,0,0,1e-13],[-1,0,1,0,0,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,-1,0,1,0,0,0,0,0,0],[0,0,0,0,-1,0,-1e-13,0,0,0,0,0],
                [0,0,0,0,0,1e-13,0,1,0,0,0,0],[0,0,0,0,0,0,-1,0,1,0,0,0],[0,0,0,0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,-1,0,1],[-1e-13,0,0,0,0,0,0,0,0,0,-1,0]])
#Hop = np.array([[0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0]])
Hop = np.zeros((12,12), dtype=complex)
Hop[2][3], Hop[3][2], Hop[8][9], Hop[9][8] = -t, -t, -t, -t

A = np.kron(I_half_a,Da) + np.kron(I_half_b, Db) + np.kron(SOC, T) + np.kron(Hop, I2)
b = np.zeros((24,1))
trial_sol = np.array([[-0.1529j, 0.1529j,0.1735j, -0.1735j,-0.1396j, 0.1396j, 0.1396, 0.1396,-0.1735, -0.1735,0.1529, 0.1529,0.1614j, -0.1614j,-0.1831j, 0.1831j,0.1473j, -0.1473j,-0.1473, -0.1473,0.1831, 0.1831,-0.1614, -0.1614
]])

trial_result = (A @ trial_sol.T)
wanted_result = (0.03544050999704 * trial_sol.T)

print(trial_result)
#print("\n", wanted_result)
for i in range(len(trial_result)//2):
    density = np.abs(trial_result[2*i][0])**2 + np.abs(trial_result[2*i+1][0])**2
    print(f"{density:.4f}", end=", ")
print("\n\n\n")
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
for i, d in enumerate(density): print(rf"$|e_{i}|^2$ = {d}")

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
