import sys

import numpy as np
import matplotlib.pyplot as plt
from system_model import System
from diamond_model import Diamond
from plot_data import DataPlot

# 해당 파일 실행 방법: python plot_xi.py (vary할 변수) (변수 시작값) (변수 끝값) (개수)
# ex) python plot_xi.py lda -4 2 7 (lda는 0.1*2^p; power로 환산)

var_param, bounds, count = (sys.argv[1], [float(x) for x in sys.argv[2:4]],
                            int(sys.argv[4]))  # ex) python plot_band_gap.py size 15 64 50
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

def get_xi(variation_values):
    global var_param, size_bounds, size_count, lda_power_bounds, lda_count, model

    xi_list = []
    xi_err_list = []
    for value in variation_values:
        new_params = params.copy()

        if var_param == 'size':
            new_params[5:7] = [float(value), float(value)]
        else:
            new_params[param_idx[var_param]] = value

        model = Diamond(a, new_params)

        model.load_eigen_data()
        model.set_corner_states()

        cstate = model.corner_states[0]
        rsd_data = model.get_rsd_data([cstate], path=1.5)

        x, density_list, _, xlabels, _, path, _ = rsd_data
        sites, density = xlabels, density_list[0]

        # print(density_list[0])
        with open(f"organized_data/1D_density/{model.filename}(state#{cstate}).txt", "w") as f:
            f.write(f"path: {path}\n{'site#':<10}{'Density':<35}\n")
            for site in range(len(sites)):
                f.write(f"{sites[site]:<10.0f}{density[site]:<35}\n")
            f.close()

        x_fit = x[:len(x) // 2]
        d_fit = density_list[0][:len(density_list[0]) // 2]

        parameters, param_errors, x_fit, d_fit = model.get_exp_dissolve_fit(x_fit, d_fit, log=True,
                                                                            return_log=False)
        A, B, xi, C = parameters
        A_err, B_err, xi_err, C_err = param_errors

        with open(f"organized_data/1D_density_fit/{model.filename}(state#{cstate}).txt", "w") as f:
            f.write(f"xi={xi}\n")
            f.write(f"path: {path}\n{'site#':<10}{'Density':<35}\n")
            for i in range(len(x_fit)):
                f.write(f"{x_fit[i]:<10.0f}{d_fit[i]:<35}\n")
            f.close()

        xi_list.append(xi)
        xi_err_list.append(xi_err)

    return xi_list, xi_err_list

if __name__=="__main__":
    variation_values = get_variation_values()
    if var_param == "size":
        fix_param_label = fr"$\lambda$={params[3]}"
        fix_param = f"lda={params[3]}"
        variation_values = [int(value) for value in variation_values]
    else:
        fix_param = f"size=({int(params[5])},{int(params[6])})"
        fix_param_label = fix_param

    xi_list, xi_err_list = get_xi(variation_values)
    print("xi list =", xi_list)
    print("xi error list =", xi_err_list)

    xi_data = variation_values, xi_list, var_param, r"$\xi$", fix_param_label, fix_param, None

    parameters, param_errors, lda_fit, xi_fit \
        = System.get_power_fit(variation_values, xi_list, log=False,return_log=False)

    A, alpha = parameters

    fig, ax = plt.subplots()
    ax.plot(lda_fit, xi_fit, color='green', label=f"{A:.4f}lda^{alpha:.4f}")
    DataPlot.draw_any_data(model, xi_data, title="Localization length", ax=ax,
                           xlog=True, ylog=True, error_bar=True,
                           y_error=xi_err_list, x_inverse=False)