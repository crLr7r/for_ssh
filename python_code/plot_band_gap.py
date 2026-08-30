import sys

import numpy as np
import matplotlib.pyplot as plt
from system_model import System
from diamond_model import Diamond
from plot_data import DataPlot

# 해당 파일 실행 방법: python plot_band_gap.py (vary할 변수) (변수 시작값) (변수 끝값) (개수)
# ex) python plot_band_gap.py size 15 64 50

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
        variation_values = [0.1 * (2 ** p) for p in power]
    else:
        variation_values = []

    return variation_values


def get_band_gap(variation_values):
    global var_param, size_bounds, size_count, lda_power_bounds, lda_count, model

    band_gap_list = []

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

        band_gap_list.append(model.evals_list[0][states[1] + 1] - model.evals_list[0][states[0] - 1])

    return band_gap_list

if __name__=="__main__":
    variation_values = get_variation_values()
    if var_param == "size":
        fix_param_label = fr"$\lambda$={params[3]}"
        fix_param = f"lda={params[3]}"
        variation_values = [int(value) for value in variation_values]
    else:
        fix_param = f"size=({int(params[5])},{int(params[6])})"
        fix_param_label = fix_param

    band_gap_list = get_band_gap(variation_values)
    band_gap_data = variation_values, band_gap_list, var_param, "Band Gap (E/t)", fix_param_label, fix_param, [0, 0.367043]
    DataPlot.draw_any_data(Diamond,band_gap_data, title="Band Gap", xlog=False, ylog=False, x_inverse=True)