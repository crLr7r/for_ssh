import numpy as np
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
    model_name, variation_param = (sys.argv[1], sys.argv[2])

    if model_name == 'Diamond':
        #Diamond 1 0 1 0.1 0.2 1 30 31
        from diamond_model import Diamond
        a, Delta, t, t_SO, lda, B, n, m = 1, 0, 1, 0.1, 0.2, 1, 30, 31      # 디폴트 값
        if variation_param == 't_SO':
            for t_SO in np.arange(0, t, 0.1):
                params = Delta, t, t_SO, lda, B, n, m
                model = Diamond(a, params)
                save_eigen_data(model)
        elif variation_param == 'lda':
            for lda in np.arange(0, t, 0.1):
                params = Delta, t, t_SO, lda, B, n, m
                model = Diamond(a, params)
                save_eigen_data(model)
        elif variation_param == 'size':
            for n in np.arange(4, 50, 1):
                m = n
                params = Delta, t, t_SO, lda, B, n, m
                model = Diamond(a, params)
                save_eigen_data(model)

    if model_name == 'Ribbon':
        #Ribbon 1 0 1 0.1 0.2 1 55
        from ribbon_model import Ribbon
        a, Delta, t, t_SO, lda, B, n = 1, 0, 1, 0.1, 0.2, 1, 55  # 디폴트 값
        if variation_param == 't_SO':
            for t_SO in np.arange(0, t, 0.1):
                params = Delta, t, t_SO, lda, B, n
                model = Diamond(a, params)
                save_eigen_data(model)
        elif variation_param == 'lda':
            for lda in np.arange(0, t, 0.1):
                params = Delta, t, t_SO, lda, B, n
                model = Diamond(a, params)
                save_eigen_data(model)
        elif variation_param == 'size':
            for n in np.arange(0, 55, 1):
                params = Delta, t, t_SO, lda, B, n
                model = Diamond(a, params)
                save_eigen_data(model)

    if model_name == 'Bulk':
        #Bulk 1 0 1 0.1
        from bulk_model import Bulk
        a, params = 1, 0, 1, 0.1
        for k_y in np.linspace(0, 2*np.pi/(np.sqrt(3)*a) , 3):
            model = Bulk(a, params, k_y=k_y)
            save_eigen_data(model)

    #
