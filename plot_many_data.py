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
        params = [0, 1, 0.1, 0.2, 1, 30, 31]
        #         Δ  t  tSO  lda B  n   m

        param_idx = {
            't_SO': 2,
            'lda': 3
        }

        if variation_param == 'size':
            power = range(6, 1, -1)
            variation_values = []
            for p in power: variation_values.append(2 ** p)
        else:
            variation_values = np.arange(0, params[1], 0.1)

        xi_list = []
        band_gap_list = []
        split_list = []

        for value in variation_values:
            new_params = params.copy()

            if variation_param == 'size':
                new_params[5:7] = [value, value]
            else:
                new_params[param_idx[variation_param]] = round(value,1)

            model = Diamond(a, new_params)
            model.load_eigen_data()
            model.set_corner_states()

            states = model.corner_states
            rsd_data = model.get_rsd_data([states[0]], path=1)

            x, density_list, *_ = rsd_data
            '''
            x_fit = x[:len(x)//2]
            d_fit = density_list[0][:len(density_list[0])//2]

            A, B, xi, C = model.get_exp_fit(x_fit, d_fit, log=True)
            xi_list.append(xi + B)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                x_fit,
                A * np.exp(-np.abs(x_fit - B) / xi),
                color='green'
            )
            '''
            band_gap_list.append(model.evals_list[0][states[1] + 1] - model.evals_list[0][states[0] - 1])
            split_list.append(model.evals_list[0][states[1]] - model.evals_list[0][states[0]])
            #DataPlot.draw_rsd(model, rsd_data, ax=ax)

        #DataPlot.draw_any_data(model,(variation_values[1:-1], xi_list[1:-1], variation_param, r"$\xi$", f"n={n}, m={m}", None), title="Localization length")

        DataPlot.draw_any_data(model,(variation_values, band_gap_list, variation_param, "Band Gap (E/t)", f"$\lambda$={params[3]}", [0, 0.367043]), title="Band Gap", log=True)

        #DataPlot.draw_any_data(model, (variation_values[26:], split_list[26:], variation_param, "Splitting (E/t)", f"$\lambda$={params[3]}", None), title="Splitting")
        
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