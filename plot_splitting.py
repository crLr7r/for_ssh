import sys

import numpy as np
import matplotlib.pyplot as plt
from system_model import System
from diamond_model import Diamond
from plot_data import DataPlot

# 해당 파일 실행 방법: python plot_splitting.py (vary할 변수) (변수 시작값) (변수 끝값) (개수)
# ex) python plot_splitting.py size 15 64 50

var_param, bounds, count = (sys.argv[1], [float(x) for x in sys.argv[2:4]], int(sys.argv[4]))
# default
a = 1
params = [0.0, 1.0, 0.1, 0.2, 1.0, 30.0, 30.0]
#         Δ    t    tSO  lda  B    n     m
n, m = int(params[5]), int(params[6])
param_idx = {
    't_SO': 2,
    'lda': 3
}

size_bounds = bounds
size_count = count

lda_power_bounds = bounds
lda_count = count

model = None


def get_variation_values():
    global var_param, size_bounds, size_count, lda_power_bounds, lda_count, model

    if var_param == 'size':
        variation_values = [float(int(v)) for v in np.linspace(size_bounds[0], size_bounds[1], size_count)]
        '''
        power = range(6, 3, -1)
        variation_values = []
        for p in power: variation_values.append(float(2 ** p))
        '''
    elif var_param == 'lda':
        power = np.linspace(lda_power_bounds[0], lda_power_bounds[1], lda_count)
        variation_values = [0.1 * (2 ** int(p)) for p in power]
    else:
        variation_values = []

    return variation_values

def get_splitting(variation_values):
    global var_param, size_bounds, size_count, lda_power_bounds, lda_count, model
    split_list = []

    for value in variation_values:
        new_params = params.copy()

        if var_param == 'size':
            new_params[5:7] = [float(value), float(value)]
        else:
            new_params[param_idx[var_param]] = value

        model = Diamond(a, new_params)

        model.load_eigen_data()
        model.set_corner_states()

        states = model.corner_states

        split_list.append(model.evals_list[0][states[1]] - model.evals_list[0][states[0]])

    return split_list

if __name__=="__main__":
    variation_values = get_variation_values()
    if var_param=="size":
        fix_param = f"lda={params[3]}"
        fix_param_label = fr"$\lambda$={params[3]}"
        variation_values = [int(value) for value in variation_values]
    else:
        fix_param = f"size=({int(params[5])},{int(params[6])})"
        fix_param_label = fix_param

    split_list = get_splitting(variation_values)

    split_data = variation_values, split_list, var_param, "Splitting (E/t)", fix_param_label, fix_param, None

    # 피팅 ------------------------------
    slope, intercept = np.polyfit(variation_values, np.log(split_list), 1)

    xi = -1 / slope
    A = np.exp(intercept)
    split_fit = A * np.exp(-np.asarray(variation_values) / xi)

    fig, ax = plt.subplots()
    ax.plot(variation_values, split_fit, color='green', label=f"xi={xi}")

    # 원본 데이터 그리기 -------------------
    DataPlot.draw_any_data(Diamond, split_data, title="Splitting", ax=ax, xlog=False, ylog=True, base=np.e, x_inverse=False)
