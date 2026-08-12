import numpy as np
from diamond_model import Diamond
from ribbon_zeeman_model import Ribbon_zeeman

import sys


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

    np.savez(
        model.filename,
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
        model = Diamond(a, params)

    if model_name == 'Ribbon_zeeman':
        #Ribbon_zeeman 1 0 1 0.1 0.2 1 55
        model = Ribbon_zeeman(a, params)

    save_eigen_data(model)
