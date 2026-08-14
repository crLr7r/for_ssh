import sympy as sp

(a0_up, a0_down, a1_up, a1_down, a2_up, a2_down, a3_up, a3_down,
b0_up, b0_down, b1_up, b1_down, b2_up, b2_down, b3_up, b3_down) = sp.symbols('a0_up a0_down a1_up a1_down a2_up a2_down a3_up a3_down b0_up b0_down b1_up b1_down b2_up b2_down b3_up b3_down', real=False)

elts = [a0_up, a0_down, a1_up, a1_down, a2_up, a2_down, a3_up, a3_down, b0_up, b0_down, b1_up, b1_down, b2_up, b2_down, b3_up, b3_down]

Del, lam, t_SO, t = sp.symbols('Del lam t_SO t', real=True)

D = sp.Matrix([[Del, -sp.I*lam], [sp.I*lam, Del]])
D2 = sp.Matrix([[-Del, -sp.I*lam], [sp.I*lam, -Del]])
T = sp.Matrix([[sp.I*t_SO, 0], [0, -sp.I*t_SO]])
I2 = sp.eye(2)

I_half1, I_half2 = sp.eye(8), sp.eye(8)
for i in range(4):
    I_half2[i, i] = 0
    I_half1[i+4, i+4] = 0

SOC = sp.Matrix([[0,1,0,0,0,0,0,0], [-1,0,1,0,0,0,0,0],[0,-1,0,1,0,0,0,0],[0,0,-1,0,0,0,0,0], [0,0,0,0,0,-1,0,0], [0,0,0,0,1,0,-1,0],[0,0,0,0,0,1,0,-1],[0,0,0,0,0,0,1,0]])
Hop = sp.Matrix([[0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0]])

A = sp.kronecker_product(I_half1,D) + sp.kronecker_product(I_half2, D2) + sp.kronecker_product(SOC, T) + sp.kronecker_product(Hop, -t*I2)
b = sp.zeros(16, 1)

solution_set = sp.linsolve((A, b), (a0_up, a0_down, a1_up, a1_down, a2_up, a2_down, a3_up, a3_down, b0_up, b0_down, b1_up, b1_down, b2_up, b2_down, b3_up, b3_down))
solution = next(iter(solution_set))

a1_a0 = sp.Abs(solution[3] + solution[2])**2 - sp.Abs(solution[1] + solution[0])**2

func = sp.lambdify((Del, lam, t_SO), a1_a0)

param_vals = 0, 0.2, 0.1
#print(func(*param_vals))

print(solution)
#print(a1_a0)
