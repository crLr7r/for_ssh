import numpy as np
import matplotlib.pyplot as plt
from system_model import System
import sys
from diamond_model import Diamond

# 해당 파일 실행 방법: python organize_many_data.py (어떤 그래프를 그릴건지)
# ex) python organize_many_data.py band_gap / python organize_many_data.py splitting / python organize_many_data.py xi

if __name__ == '__main__':
    
    graph = sys.argv[1]

    # default
    a = 1
    params = [0.0, 1.0, 0.1, 0.2, 1.0, 30.0, 30.0]
    #         Δ    t    tSO  lda  B    n     m
    n,m = int(params[5]), int(params[6])
    param_idx = {
        't_SO': 2,
        'lda': 3
    }
    
    size_bounds = [32, 64]
    size_count = 10

    lda_power_bounds = [-4, 2]

    def get_variation_values(var_param):
        
        if var_param == 'size':
            variation_values = [float(int(v)) for v in np.linspace(size_bounds[0], size_bounds[1], size_count)]
            '''
            power = range(6, 3, -1)
            variation_values = []
            for p in power: variation_values.append(float(2 ** p))
            '''
        elif var_param == 'lda':
            power = range(lda_power_bounds[0], lda_power_bounds[1] + 1)
            variation_values = [0.1*(2**p) for p in power]
        else:
            variation_values = np.arange(0, 0.6, 0.1)
        
        return variation_values

    def get_band_gap(variation_values, var_param):
        
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
    
    def get_splitting(variation_values, var_param):
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

    def get_xi(variation_values, var_param):
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
    
            #print(density_list[0])
            with open(f"organized_data/1D_density/{model.filename}(state#{cstate}).txt", "w") as f:
                f.write(f"path: {path}\n{'site#':<10}{'Density':<35}\n")
                for site in range(len(sites)):
                    f.write(f"{sites[site]:<10.0f}{density[site]:<35}\n")
                f.close()
            
            x_fit = x[:len(x)//2]
            d_fit = density_list[0][:len(density_list[0])//2]

            parameters, param_errors, x_fit, d_fit = model.get_exp_dissolve_fit(x_fit, d_fit, log=True, return_log=False)
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
            
            
    # Band gap - size --------------------------------------------------------------------------
    if graph == 'band_gap':
        sizes = get_variation_values("size")
        sizes = [int(value) for value in sizes]
        band_gap_list = get_band_gap(sizes, "size")
        
        with open(f"organized_data/Bandgap_size/Diamond band gap(size=[{size_bounds[0],size_bounds[1]}], count={size_count}).txt", "w") as f:
            f.write(f"{'size(n)':<10}{'Band Gap (E/t)':<30}")
            f.write(f"lambda={params[3]}\n")
            for i, size in enumerate(sizes):
                f.write(f"{size:<10}{band_gap_list[i]:<30}\n")
            f.close()
       
    # Splitting - size -------------------------------------------------------------------------
    if graph == 'splitting':
        sizes = get_variation_values("size")
        sizes = [int(value) for value in sizes]
        split_list = get_splitting(sizes, "size")

        with open(f"organized_data/Splitting_size/Diamond splitting(size=[{size_bounds[0],size_bounds[1]}], count={size_count}).txt", "w") as f:
            f.write(f"{'size(n)':<10}{'Splitting (E/t)':<30}")
            f.write(f"lambda={params[3]}\n")
            for i, size in enumerate(sizes):
                f.write(f"{size:<10}{split_list[i]:<30e}\n")
            f.close()

        slope, intercept = np.polyfit(sizes, np.log(split_list), 1)

        xi = -1 / slope
        A = np.exp(intercept)
        split_fit = A * np.exp(-np.asarray(sizes) / xi)
        
        with open(f"organized_data/Splitting_size_fit/Diamond splitting fit(size=[{size_bounds[0],size_bounds[1]}], count={size_count}).txt", "w") as f:
            f.write(f"xi={xi}\n")
            f.write(f"{'size(n)':<10}{'Splitting fit (E/t)':<30}")
            f.write(f"lambda={params[3]}\n")
            for i, size in enumerate(sizes):
                f.write(f"{size:<10}{split_fit[i]:<30e}\n")
            f.close()

    # xi - lamda ---------------------------------------------------------------------------------
    if graph == 'xi':
        ldas = get_variation_values("lda")
        xi_list, xi_err_list = get_xi(ldas, "lda")

        with open(f"organized_data/xi_lda/Diamond xi(lda=[0.1x2^{lda_power_bounds[0]}, 0.1x2^{lda_power_bounds[1]}]).txt", "w") as f:
            f.write(f"{'lda':<15}{'xi':<30}{'error':<20}\n")
            for i, lda in enumerate(ldas):
                f.write(f"{lda:<15.5f}{xi_list[i]:<30.16f}{xi_err_list[i]:<20}\n")

        parameters, param_errors, lda_fit, xi_fit= System.get_power_fit(ldas, xi_list, log=False, return_log=False)
        A, alpha = parameters

        with open(f"organized_data/xi_lda_fit/Diamond xi fit(lda=[0.1x2^{lda_power_bounds[0]}, 0.1x2^{lda_power_bounds[1]}]).txt", "w") as f:
            f.write(f"{A:.4f}lambda^{alpha:.4f}\n")
            f.write(f"{'lda':<15}{'xi':<30}{'error':<20}\n")
            for i, lda in enumerate(ldas):
                f.write(f"{lda:<15.5f}{xi_list[i]:<30.16f}{xi_err_list[i]:<20}\n")