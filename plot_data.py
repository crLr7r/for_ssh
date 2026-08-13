import numpy as np
from system_plot import DataPlot
from diamond_model import Diamond
from ribbon_zeeman_model import Ribbon_zeeman

import sys

if __name__ == '__main__':
    model_name, a, params = (sys.argv[1],
                                 float(sys.argv[2]),
                                 [float(x) for x in sys.argv[3:]])

    if model_name == 'Diamond':
        # Diamond 1 0 1 0.1 0.2 1 30 31 (a Delta t t_SO lda B n m)
        model = Diamond(a, params)
        is_kspace = False
        model.load_eigen_data()
        model.set_corner_states()

        # 가변적 data ---------------------------------------------------------
        states = model.corner_states
        
        band_data = model.get_band_data()
        rsd_data = model.get_rsd_data([states[1]], path=1)
        rsd_2D_data = model.get_2D_rsd_data([states[1]])
        
        print(f"band gap = {model.evals_list[0][states[1] + 1] - model.evals_list[0][states[0] - 1]:.14f}")
        print(f"E1={model.evals_list[0][states[0]]:.14f}, E2={model.evals_list[0][states[1]]:.14f}")
        
        e = [(1,0,3),(1, 0, 2), (1, 0, 1), (0,0,1), (1, 0, 0), (0, 0, 0), (1,1,0),(0, 1, 0), (0, 2, 0), (0,3,0)]
        for i, s in enumerate(e):
            site = [2*model.s_comp(*s), 2*model.s_comp(*s)+1]
            elt = [model.evecs_list[0][states[0]][site[0]], model.evecs_list[0][states[0]][site[1]]]
            print(f"e{i+1} = ({elt[0]:.4f}, {elt[1]:.4f})")
            print(f"e{i+1}^2 = {np.abs(elt[0])**2 + np.abs(elt[1])**2:.4f}")


        # plot ---------------------------------------------------------------
        '''
        DataPlot.draw_band(model, band_data, is_E_bounded=True, is_x_bounded=True, is_kspace=is_kspace)
        DataPlot.draw_rsd(model, rsd_data, log=False)
        DataPlot.draw_2D_rsd(model, rsd_2D_data)
        '''

    elif model_name == 'Ribbon_zeeman':
        #Ribbon_zeeman 1 0 1 0.1 0.2 1 55
        model = Ribbon_zeeman(a, params)
        is_kspace = True
        model.load_eigen_data()

        # 가변적 data ---------------------------------------------------------
        ver_list = [1, 2, 3, 4]
        band_data = model.get_HL_band_data(ver_list, distinct="spin")
        Elist = []
        for E in model.evals_list[model.KPOINTS // 2]:
            if np.isclose(E, 0, atol=0.25, rtol=0.1):
                Elist.append(E)
        print(f"ribbon_zeeman band gap = {Elist[2] - Elist[1]:.6f}")
        '''
        model.set_xi_data(ver=1)    # 돌아가는 데 비교적 오래 걸리므로 필요할 때만 실행할 것
        xi_list, _, _ = model.xi_data

        max_idx = [np.argmax(xi_list[:model.KPOINTS // 3]),
                   np.argmax(xi_list[2 * model.KPOINTS // 3:]) + 2 * model.KPOINTS // 3]
        states = [model.klist[i] for i in np.arange(max_idx[0], max_idx[1] + 1, (max_idx[1] - max_idx[0])//5)]

        rsd_data = model.get_rsd_data(states, path="basis")
        '''
        # plot ---------------------------------------------------------------
        DataPlot.draw_band(model, band_data, is_E_bounded=True, is_x_bounded=True, is_kspace=is_kspace)
        #DataPlot.draw_rsd(model, rsd_data, log=False)
        #DataPlot.draw_kspace_val(model, model.xi_data)

