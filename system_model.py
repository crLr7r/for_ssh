import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.optimize import curve_fit

class System(ABC):
    # 클래스 변수 -----------------------------------------------------------
    KPOINTS = 501
    MAX_TICKS = 16

    param_names = None
    name = None

    # 인스턴스 함수 정의--------------------------------------------------------
    @abstractmethod
    def __init__(self, a, params):  # params과 param_vals은 일대일 대응

        self.a = a
        self.params = params

        self.kmin, self.kmax = 0, 0
        self.klist = self.get_klist()

        self.basis_num = None
        self.filename = None

    def get_klist(self):
        return np.linspace(self.kmin, self.kmax, self.KPOINTS)

    def set_kspace_data(self):
        self.klabel = None
        self.line_indicator = None
        self.spec_indices = None

    # 그래프나 사진 이름 지정
    def set_title(self, title=None):
        if title is not None:
            self.title = title
        else:
            # parameter들 값 명시
            param_string = [f'a={self.a}']
            for i, p in enumerate(self.param_names):
                param_string.append(str(p) + "=" + str(self.params[i]))
            self.title = ", ".join(param_string)


    @abstractmethod
    def H(self, k, k_y=0):
        pass

    def load_eigen_data(self):
        with np.load(f"data/{self.filename}") as data:
            self.evals_list = data["evals_list"]
            self.evecs_list = data["evecs_list"]

    # 어떤 state에 대한 total density를 리턴하는 함수(1인지 확인용)
    def get_total_density(self, state, k=0, do_print=True):
        total_density = np.linalg.norm(self.evecs_list[k][state])
        if do_print: print(f"total_density = {total_density}")
        return total_density

    @abstractmethod
    def get_band_data(self):
        pass


    def get_rsd_data(self, idx_list, path=None):
        pass


    def get_2D_rsd_data(self, idx_list, spin=None):
        pass

     # 어떤 xlist, dlist 데이터에 대해 지수 감쇠 모델을 피팅하는 함수
    def get_exp_fit(self, xlist, dlist, log=True):  # xi_0: 초기 추정값
        x = np.asarray(xlist)
        y = np.asarray(dlist)

        if y[-1] > y[0]: y = y[::-1]  # 반대 그래프의 경우 순서 뒤집기

        # 피팅할 함수: 지수 감쇠 함수
        def exp_dissolve(x, A, B, xi, C):
            return A * np.exp(-np.abs(x - B) / xi) + C

        # 피팅할 함수2: 세미로그 함수
        def exp_log_dissolve(x, A, B, xi, C):
            return np.log(A) - np.abs(x - B) / xi

        # 초기 추정값
        C0 = np.min(y)
        B0 = x[np.argmax(y)]
        xi0 = (x[-1] - x[0]) / 5
        A0 = np.max(y) - C0

        # 로그 스케일일 경우
        fit_func = exp_dissolve
        if log:
            y = np.log(np.maximum(y, 1e-13))
            fit_func = exp_log_dissolve
            for idx in range(len(y)):
                if idx == 0: continue
                if np.isclose(y[idx], y[idx - 1], rtol=1e-8, atol=1e-10):
                    y = y[:idx]
                    x = x[:idx]
                    break

        # 피팅
        parameters, _ = curve_fit(
            fit_func,
            x,
            y,
            p0=[A0, B0, xi0, C0],  # A, B, xi, C의 초기 추정값
            bounds=([0, 0, 1e-10, 0], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=100000
        )

        return parameters

