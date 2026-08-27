import numpy as np
from diamond_model import Diamond
import matplotlib.pyplot as plt
from system_plot import DataPlot
from system_model import System

# 시스템 입력 변수 따로 없음
'''
size = [4, 8, 16, 32, 64]
run_time = [260, 380, 1053, 41419, 2063174]
'''

size = [16, 32, 64]
run_time = [1053, 41419, 2063174]

size_inverse = 1 / np.asarray(size)
size_inverse_log = [np.log2(si) for si in size_inverse]
log_run_time = [np.log2(rt) for rt in run_time]

params, param_errors, size_fit, runtime_fit \
    = System.get_power_fit(size, run_time, log=False, return_log=False)

print(params)
A, alpha = params

print(f"Run time = {A:.4f}n^{alpha:.4f}")
print("Run time ( n = 90 ) = ", System.power(90, A, alpha))

fig, ax = plt.subplots()
ax.plot(1 / np.asarray(size_fit), runtime_fit, color='green', label=f"{A:.4f}n^{alpha:.4f}")

run_time_data = (size, run_time, 'size', "Running time(ms)", r"$\lambda$=0.2", "lda=0.2", None)
DataPlot.draw_any_data(Diamond, run_time_data,
                       title="Running Time", ax=ax, xlog=True, ylog=True, x_inverse=True)