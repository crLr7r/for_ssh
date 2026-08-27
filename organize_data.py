import numpy as np
from system_model import System
from diamond_model import Diamond
from ribbon_model import Ribbon
import sys

model_name, a, params = (sys.argv[1],
                                 float(sys.argv[2]),
                                 [float(x) for x in sys.argv[3:]])      # python organize_data.py Diamond 1 0 1 0.1 0.2 1 30 30 <<이런식으로 입력하면 됨

if model_name == "Diamond":
    # Diamond 1 0 1 0.1 0.2 1 30 31 (a Delta t t_SO lda B n m)
    model = Diamond(a, params)
    is_kspace = False
    model.set_title()
    
    # 1. Diamond eigenvalue -------------------------------------------------------------------------------
    model.load_eigen_data()
    #print(model.evals_list[0])
    with open(f"organized_data/Eigenvalue/{model.filename}.txt", "w") as f:
        f.write(f"{'state#':<10}{'Energy':<20}\n")
        for state in model.state_list:
            f.write(f"{state:<10}{model.evals_list[0][state]:<20}\n")
        f.close()

    # 2. Diamond cornerstate eigenvalue & eigenvector -----------------------------------------------------
    model.set_corner_states()
    cstate = model.corner_states[0]
    with open(f"organized_data/Corner_states_data/{model.filename}(state#{cstate}).txt", "w") as f:
        f.write(f"state#={cstate:<10}\nE={model.evals_list[0][cstate]:<20}\n")
        for i in model.state_list:
            elt = model.evecs_list[0][cstate][i]
            f.write(f"{elt:<28e}     ")
            if i % 2 == 1: f.write("\n")
        f.close()

    # 3. Density(1D) : a-site half edge(corner state에 대해) -----------------------------------------------
    cstate = model.corner_states[0]
    rsd_data = model.get_rsd_data([cstate], path=1.5)
    _, density_list, _, xlabels, _, path, _ = rsd_data
    sites, density = xlabels, density_list[0]
    
    #print(density_list[0])
    with open(f"organized_data/1D_density/{model.filename}(state#{cstate}).txt", "w") as f:
        f.write(f"path: {path}\n{'site#':<10}{'Density':<35}\n")
        for site in range(len(sites)):
            f.write(f"{sites[site]:<10.0f}{density[site]:<35}\n")
        f.close()

    # 4. Density(2D) : corner state에 대한 density ---------------------------------------------------------
    cstate = model.corner_states[1]
    rsd_2D_data = model.get_2D_rsd_data([cstate])
    x, y, d_func, _, _, _ = rsd_2D_data

    X, Y = np.meshgrid(x, y)
    
    with open(f"organized_data/2D_density/{model.filename}(state#{cstate}).txt", "w") as f:
        f.write(f"{'x':<25}{'y':<10}{'weight':<30}\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i, j], Y[i, j]
                f.write(f"{x:<25.16f}{y:<10}{d_func(x, y):<30e}\n")
        f.close()



if model_name == "Ribbon":
    #Ribbon 1 0 1 0.1 0.2 1 55 (a Delta t t_SO lda B n)
    model = Ribbon(a, params)
    is_kspace = True
    model.load_eigen_data()
    
    Elist = []
    for E in model.evals_list[model.KPOINTS // 2]:
        if np.isclose(E, 0, atol=0.25, rtol=0.1):
            Elist.append(E)
    
    with open(f"organized_data/Ribbon_band_gap/{model.filename}.txt", "w") as f:
        f.write(f"{Elist[2] - Elist[1]}")


        
        