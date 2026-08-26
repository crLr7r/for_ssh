import numpy as np
from system_plot import DataPlot
import matplotlib.pyplot as plt

import sys

if __name__ == '__main__':
    model_name, a, params = (sys.argv[1],
                                 float(sys.argv[2]),
                                 [float(x) for x in sys.argv[3:]])      # Diamond 1 0 1 0.1 0.2 1 30 30 <<이런식으로 입력하면 됨

    if model_name == 'Diamond':
        # Diamond 1 0 1 0.1 0.2 1 30 31 (a Delta t t_SO lda B n m)
        from diamond_model import Diamond
        model = Diamond(a, params)
        is_kspace = False
        model.load_eigen_data()
        model.set_corner_states()
        ax = None
        # 가변적 data ---------------------------------------------------------
        states = model.corner_states
        n,m = model.params[5], model.params[6]

        '''
        #대각화 잘 됐는지 체크
        LHS = model.H(0) @ model.evecs_list[0][states[0]]
        RHS = model.evals_list[0][states[0]] * model.evecs_list[0][states[0]]
        print(RHS-LHS, "\n\n\n")
        '''
        band_data = model.get_band_data()
        rsd_data = model.get_rsd_data([states[1]], path=1)
        rsd_2D_data = model.get_2D_rsd_data([states[1]])
        
        xlist, dlist, _, _, _, _, _ = rsd_data
        xlist = xlist[:len(xlist)//2]
        dlist = dlist[0][:len(dlist[0])//2]
        parameters, param_erros, _, _ = model.get_exp_dissolve_fit(xlist, dlist, log=True)
        A,B,xi,C=parameters
        
        t_SO, lda = model.params[2], model.params[3]
        print(f"t_SO={t_SO}, lda={lda}")
        print(f"numerical xi={xi}")
        k = lda/t_SO
        alpha = -(k/2) + np.sqrt((k/2) ** 2 + 1)
        analytic_xi = -1/(2*np.log(alpha))
        print(f"analytic xi={analytic_xi}")

        fig, ax = plt.subplots(figsize=(10, 5))
        #ax.plot(xlist,A * np.exp(-np.abs(xlist - B) / xi),color='green')

        #print(f"band gap(n={n}, m={m}) = {model.evals_list[0][states[1] + 1] - model.evals_list[0][states[0] - 1]:.14f}")
        #print(f"E1={model.evals_list[0][states[0]]:.14f}, E2={model.evals_list[0][states[1]]:.14f}")
        model.get_total_density(states[0], do_print=False)

        state_num = states[1] -1000
        extra = [(1, 1, 1), (0, 1, 1)]
        a = [(0,0,0), (0,1,0), (0,2,0)]
        b = [(1,0,0), (1,0,1), (1,0,2)]
        a_right = [(0,n-1,m-1), (0,n-1,m-2), (0,n-1,m-3)]
        b_right = [(1,n-1,m-1), (1,n-2,m-1), (1,n-3,m-1)]
        #e = [(1, 0, 3), (1, 0, 2), (1, 0, 1), (0, 0, 1), (1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]
        density_list = []
        e_list = []
        
        #print("corner state #",state_num, "\n")
        print("Bulk state #",state_num, "\n")
        for i, s in enumerate(a):
            site = [2*model.s_comp(*s), 2*model.s_comp(*s)+1]
            elt = [model.evecs_list[0][state_num][site[0]], model.evecs_list[0][state_num][site[1]]]
            print(f"a{i+1} = ({elt[0]:.10f}, {elt[1]:.10f})")
            density = np.abs(elt[0])**2 + np.abs(elt[1])**2
            density_list.append(density)
            for elt_i in elt: e_list.append(elt_i)
        print()
        for i, s in enumerate(b):
            site = [2*model.s_comp(*s), 2*model.s_comp(*s)+1]
            elt = [model.evecs_list[0][state_num][site[0]], model.evecs_list[0][state_num][site[1]]]
            print(f"b{i+1} = ({elt[0]:.10f}, {elt[1]:.10f})")
            density = np.abs(elt[0])**2 + np.abs(elt[1])**2
            density_list.append(density)
            for elt_i in elt: e_list.append(elt_i)
            #print(f"e{i+1}^2 = {density:.4f}")
        #for e_i in e_list: print(f"{e_i:.10f}", end=", ")
        #print(f"corner site density={density_list[5]:.4f}")
        #print(f"decay(lda={model.params[3]})={density_list[5] - density_list[7]:.15f}")
        
        # plot ---------------------------------------------------------------

        #DataPlot.draw_rsd(model, rsd_data, log=False, ax=ax)
        #DataPlot.draw_band(model, band_data, is_E_bounded=True, is_x_bounded=True, is_kspace=is_kspace)
        DataPlot.draw_2D_rsd(model, rsd_2D_data, title=r"Density($\Delta$=0, $t_{SO}$=0.1, $\lambda$=0.2, n=30, m=30)")
        

    elif model_name == 'Ribbon':
        #Ribbon 1 0 1 0.1 0.2 1 55
        from ribbon_model import Ribbon
        
        model = Ribbon(a, params)
        is_kspace = True
        model.load_eigen_data()

        # 가변적 data ---------------------------------------------------------
        ver_list = [1, 2, 3, 4]
        band_data = model.get_HL_band_data(ver_list, distinct="spin")
        Elist = []
        for E in model.evals_list[model.KPOINTS // 2]:
            if np.isclose(E, 0, atol=0.25, rtol=0.1):
                Elist.append(E)
        print(f"ribbon band gap(n={model.params[5]}) = {Elist[2] - Elist[1]:.6f}")
        '''
        model.set_xi_data(ver=1)    # 돌아가는 데 비교적 오래 걸리므로 필요할 때만 실행할 것
        xi_list, _, _ = model.xi_data

        max_idx = [np.argmax(xi_list[:model.KPOINTS // 3]),
                   np.argmax(xi_list[2 * model.KPOINTS // 3:]) + 2 * model.KPOINTS // 3]
        states = [model.klist[i] for i in np.arange(max_idx[0], max_idx[1] + 1, (max_idx[1] - max_idx[0])//5)]

        rsd_data = model.get_rsd_data(states, path="basis")
        '''
        # plot ---------------------------------------------------------------
        #ax = DataPlot.draw_band(model, band_data, is_E_bounded=True, is_x_bounded=True, is_kspace=is_kspace)
        #DataPlot.draw_rsd(model, rsd_data, log=False)
        #DataPlot.draw_kspace_val(model, model.xi_data)
    
    elif model_name == 'Bulk':
        # Bulk 1 0 1 0.1
        from bulk_model import Bulk
        ax = None
        loop = 1
        for i, k_y in enumerate(np.linspace(0, 2*np.pi/(np.sqrt(3)*a) , loop)):
            model = Bulk(a, params, k_y=k_y)
            is_kspace = True
            model.load_eigen_data()
        
            # 가변적 data ---------------------------------------------------------
            band_data = model.get_HL_band_data()
            val_data = model.get_spin_data(band_num=0)
        
            # plot ---------------------------------------------------------------
            ax = DataPlot.draw_band(model, band_data, ax=ax, is_E_bounded=False, is_x_bounded=True, is_kspace=is_kspace, is_last=i == loop - 1)
            #DataPlot.draw_kspace_val(model, val_data)
