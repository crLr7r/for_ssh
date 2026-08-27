import numpy as np
import sys
import os
from PATH import *
# 해당 파일 실행 방법: python save_data.py (시스템 이름) (a) (Delta) (t) (t_SO) (lda) (B) (n) (m)
# ex) Diamond 1 0 1 0.1 0.2 1 30 31 / Ribbon 1 0 1 0.1 0.2 1 55

def diagonalize(model, do_print=False):
    evals_list, evecs_list = [], []

    for k in model.klist:
        H_matrix = np.asarray(model.H(k), dtype=complex)
        evals, evecs = np.linalg.eigh(H_matrix)
        evecs = evecs.T  # 행/열 뒤집기
        evals_list.append(evals)
        evecs_list.append(evecs)

        if do_print:
            for eval, evec in zip(evals, evecs):
                print(f"{eval:3.5f}, {evec}")

    return np.asarray(evals_list), np.asarray(evecs_list)


def save_eigen_data(model):

    evals_list, evecs_list = diagonalize(model)
    path = f"{DATA_DIR}/{model.filename}.npz"

    if os.path.exists(path):
        print("file already exists")

    else:
        np.savez(
            path,
            evals_list=np.asarray(evals_list),
            evecs_list=np.asarray(evecs_list),
        )

if __name__ == "__main__":
    model_name, a, params = (sys.argv[1],
                                 float(sys.argv[2]),
                                 [float(x) for x in sys.argv[3:]])

    model = None
    if model_name == 'Diamond':
        #Diamond 1 0 1 0.1 0.2 1 30 31
        from diamond_model import Diamond
        model = Diamond(a, params)
        save_eigen_data(model)

    if model_name == 'Ribbon':
        #Ribbon 1 0 1 0.1 0.2 1 55
        from ribbon_model import Ribbon
        model = Ribbon(a, params)
        save_eigen_data(model)

    if model_name == 'Bulk':
        #Bulk 1 0 1 0.1
        from bulk_model import Bulk
        for k_y in np.linspace(0, 2*np.pi/(np.sqrt(3)*a) , 3):
            model = Bulk(a, params, k_y=k_y)
            save_eigen_data(model)

    #
