import numpy as np
import sys
from PATH import DATA_DIR
# loop을 돌면서 파일을 저장하는 코드인데 시간이 넘쳐날때만 사용하기

# 해당 파일 실행 방법: python save_many_data.py (시스템명) (vary할 변수이름)
# ex) Diamond size / Diamond lda

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
        f"{DATA_DIR}/{model.filename}",
        evals_list=np.asarray(evals_list),
        evecs_list=np.asarray(evecs_list),
    )

if __name__ == "__main__":
    model_name, variation_param = (sys.argv[1], sys.argv[2])

    if model_name == 'Diamond':
        #Diamond 1 0 1 0.1 0.2 1 30 31
        from diamond_model import Diamond
        # default
        a = 1
        params = np.asarray([0, 1, 0.1, 0.2, 1, 30, 30])
        #                    Δ  t  tSO  lda B  n   m

        param_idx = {
            't_SO': 2,
            'lda': 3
        }

        if variation_param == 'size':
            power = range(6, 1, -1)
            variation_values = []
            for p in power: variation_values.append(2 ** p)
        elif variation_param == 'lda':
            power = range(-4, 1)
            variation_values = [0.1*(2**p) for p in power]
        else:
            variation_values = np.arange(0, params[1], 0.1)

        for value in variation_values:
            new_params = params.copy()

            if variation_param == 'size':
                new_params[5:7] = [value, value]
            else:
                new_params[param_idx[variation_param]] = value

            model = Diamond(a, new_params)
            save_eigen_data(model)

    if model_name == 'Ribbon':
        #Ribbon 1 0 1 0.1 0.2 1 55
        from ribbon_model import Ribbon
        # default
        a = 1
        params = np.asarray([0, 1, 0.1, 0.2, 1, 55])
        #         Δ  t  tSO  lda B  n   m

        param_idx = {
            't_SO': 2,
            'lda': 3,
            'size': 5
        }

        if variation_param == 'size':
            variation_values = range(5, 55, 5)
        else:
            variation_values = np.arange(0, params[1], 0.1)

        for value in variation_values:
            new_params = params.copy()

            new_params[param_idx[variation_param]] = round(value,1)

            model = Ribbon(a, new_params)
            save_eigen_data(model)

    if model_name == 'Bulk':
        #Bulk 1 0 1 0.1
        from bulk_model import Bulk
        a, params = 1, 0, 1, 0.1
        for k_y in np.linspace(0, 2*np.pi/(np.sqrt(3)*a) , 3):
            model = Bulk(a, params, k_y=k_y)
            save_eigen_data(model)

    #
