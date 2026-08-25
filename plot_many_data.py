import numpy as np
import matplotlib.pyplot as plt
import sys

from system_plot import DataPlot


if __name__ == '__main__':
    model_name, variation_param = sys.argv[1:3]

    if model_name == 'Diamond':
        from diamond_model import Diamond

        # default
        a = 1
        params = [0.0, 1.0, 0.1, 0.2, 1.0, 30.0, 30.0]
        #         Δ    t    tSO  lda  B    n     m
        n,m = int(params[5]), int(params[6])
        param_idx = {
            't_SO': 2,
            'lda': 3
        }

        if variation_param == 'size':
            variation_values = [float(int(v)) for v in np.linspace(15, 32, 10)]
            '''
            power = range(6, 3, -1)
            variation_values = []
            for p in power: variation_values.append(float(2 ** p))
            '''
        elif variation_param == 'lda':
            power = range(-4, 2)
            variation_values = [0.1*(2**p) for p in power]
        else:
            variation_values = np.arange(0, 0.6, 0.1)

        xi_list = []
        xi_err_list = []
        band_gap_list = []
        split_list = []

        for value in variation_values:
            new_params = params.copy()

            if variation_param == 'size':
                new_params[5:7] = [value, value]
            else:
                new_params[param_idx[variation_param]] = value

            model = Diamond(a, new_params)
            
            model.load_eigen_data()
            model.set_corner_states()

            states = model.corner_states
            rsd_data = model.get_rsd_data([states[0]], path=1.5)
            
            x, density_list, *_ = rsd_data
            
            x_fit = x[:len(x)//2]
            d_fit = density_list[0][:len(density_list[0])//2]

            parameters, param_errors, x_fit, d_fit = model.get_exp_dissolve_fit(x_fit, d_fit, log=True, return_log=True)
            A, B, xi, C = parameters
            A_err, B_err, xi_err, C_err = param_errors

            xi_list.append(xi)
            xi_err_list.append(xi_err)
            
            #fig, ax = plt.subplots(figsize=(10, 5))
            #ax.plot(x_fit,d_fit,color='green',linewidth=5)
            
            band_gap_list.append(model.evals_list[0][states[1] + 1] - model.evals_list[0][states[0] - 1])
            split_list.append(model.evals_list[0][states[1]] - model.evals_list[0][states[0]])
            #DataPlot.draw_rsd(model, rsd_data, ax=ax, log=True)
            
        if variation_param == 'size': variation_values = [int(value) for value in variation_values]
        print("xi list =", xi_list)
        print("xi error list =", xi_err_list)
       
        
        # xi ----------------------------------------------------
        xi_data = variation_values, xi_list, variation_param, r"$\xi$", f"n={n}, m={m}", None
        parameters, param_errors, lda_fit, xi_fit= model.get_power_fit(variation_values, xi_list, log=False, return_log=False)
        A, alpha = parameters
        
        
        fig, ax = plt.subplots()
        ax.plot(lda_fit, xi_fit, color='green', label=f"{A:.4f}lda^{alpha:.4f}")
        DataPlot.draw_any_data(model,xi_data, title="Localization length", ax=ax, xlog=True, ylog=True, error_bar=True, y_error=xi_err_list, x_inverse=False)
        
        # Band gap ----------------------------
        band_gap_data = variation_values, band_gap_list, variation_param, "Band Gap (E/t)", fr"$\lambda$={params[3]}", [0, 0.367043]
        #DataPlot.draw_any_data(model,band_gap_data, title="Band Gap", xlog=False, ylog=False, x_inverse=True)
        
        # Splitting -------------------------
        split_data = variation_values, split_list, variation_param, "Splitting (E/t)", fr"$\lambda$={params[3]}", None
        slope, intercept = np.polyfit(variation_values, np.log(split_list), 1)

        xi = -1 / slope
        A = np.exp(intercept)
        split_fit = A * np.exp(-np.asarray(variation_values) / xi)
       
        '''fig, ax = plt.subplots()
        ax.plot(variation_values, split_fit, color='green', label=f"xi={xi}")
        DataPlot.draw_any_data(model, split_data, title="Splitting", ax=ax, xlog=False, ylog=True, base=np.e, x_inverse=False)
        '''
        # Run time --------------------------
        '''
        size = [4, 8, 16, 32, 64]
        run_time = [260, 380, 1053, 41419, 2063174]
        '''
        size = [16, 32, 64]
        run_time = [1053, 41419, 2063174]
        
        size_inverse = 1/np.asarray(size)
        size_inverse_log = [np.log2(si) for si in size_inverse]
        log_run_time = [np.log2(rt) for rt in run_time]
        params, param_errors, size_fit, runtime_fit = model.get_power_fit(size, run_time, log=False, return_log=False)
        print(params)
        A, alpha = params

        print(f"Run time = {A:.4f}n^{alpha:.4f}")

        print("Run time ( n = 90 ) = ", model.power(90, A, alpha))

        fig, ax = plt.subplots()
        ax.plot(1/np.asarray(size_fit), runtime_fit, color='green', label=f"{A:.4f}n^{alpha:.4f}")
        
        DataPlot.draw_any_data(model, (size, run_time, 'size', "Running time(ms)", r"$\lambda$=0.2", None), 
                                        title="Running Time", ax=ax, xlog=True, ylog=True, x_inverse=True)
        
'''

if __name__ == '__main__':
    model_name, variation_param = (sys.argv[1], sys.argv[2])

    if model_name == 'Diamond':
        # Diamond 1 0 1 0.1 0.2 1 30 31 (a Delta t t_SO lda B n m)
        from diamond_model import Diamond
        is_kspace = False
        a, Delta, t, t_SO, lda, B, n, m = 1, 0, 1, 0.1, 0.2, 1, 30, 31      # 디폴트 값
        
        xlist, xi_list = None, []
        xlabel = variation_param

        if variation_param == 't_SO':
            xlist = np.arange(0, t, 0.1)
            for t_SO in xlist:
                params = [Delta, t, t_SO, lda, B, n, m]
                model = Diamond(a, params)
                model.load_eigen_data()
                model.set_corner_states()

                states = model.corner_states
        
                rsd_data = model.get_rsd_data([states[1]], path=1)
        
                xlist, dlist, _, _, _, _, _ = rsd_data
                xlist = xlist[:len(xlist)//2]
                dlist = dlist[0][:len(dlist[0])//2]
                A,B,xi,C = model.get_exp_fit(xlist, dlist, log=True)
                xi_list.append(xi)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(xlist,A * np.exp(-np.abs(xlist - B) / xi),color='green')
                DataPlot.draw_rsd(model, rsd_data, ax=ax)
                
        elif variation_param == 'lda':
            xlist = np.arange(0, t, 0.1)
            for lda in xlist:
                params = [Delta, t, t_SO, lda, B, n, m]
                model = Diamond(a, params)
                model.load_eigen_data()
                model.set_corner_states()

                states = model.corner_states
        
                rsd_data = model.get_rsd_data([states[1]], path=1)
        
                xlist, dlist, _, _, _, _, _ = rsd_data
                xlist = xlist[:len(xlist)//2]
                dlist = dlist[0][:len(dlist[0])//2]
                A,B,xi,C = model.get_exp_fit(xlist, dlist, log=True)
                xi_list.append(xi)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(xlist,A * np.exp(-np.abs(xlist - B) / xi),color='green')
                DataPlot.draw_rsd(model, rsd_data, ax=ax)
                
        elif variation_param == 'size':
            xlist = np.arange(4, 50, 1)
            for n in xlist:
                m = n
                params = [Delta, t, t_SO, lda, B, n, m]
                model = Diamond(a, params)
                model.load_eigen_data()
                model.set_corner_states()

                states = model.corner_states
        
                rsd_data = model.get_rsd_data([states[1]], path=1)
        
                xlist, dlist, _, _, _, _, _ = rsd_data
                xlist = xlist[:len(xlist)//2]
                dlist = dlist[0][:len(dlist[0])//2]
                A,B,xi,C = model.get_exp_fit(xlist, dlist, log=True)
                xi_list.append(xi)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(xlist,A * np.exp(-np.abs(xlist - B) / xi),color='green')
                DataPlot.draw_rsd(model, rsd_data, ax=ax)
        
        DataPlot.draw_localization_length(model, (xlist, xi_list, xlabel))
        
'''