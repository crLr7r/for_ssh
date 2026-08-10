import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class System(ABC):
    # 클래스 변수 -----------------------------------------------------------
    KPOINTS = 501
    MAX_TICKS = 16

    param_names = None
    name = None
    kmin = None
    kmax = None

    #클래스 함수 -----------------------------------------------------------

    @classmethod
    def get_klist(cls):
        return np.linspace(cls.kmin, cls.kmax, cls.KPOINTS)

    @classmethod
    def set_kspace_data(cls):
        cls.klabel = None
        cls.line_indicator = None
        cls.spec_indices = None


    # 인스턴스 함수 정의--------------------------------------------------------
    @abstractmethod
    def __init__(self, a, params):  # params과 param_vals은 일대일 대응

        self.a = a
        self.params = params
        self.klist = self.get_klist()

        self.basis_num = None
        self.filename = None

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
    def H(self, k):
        pass

    def load_eigen_data(self):
        with np.load(self.filename) as data:
            self.evals_list = data["evals_list"]
            self.evecs_list = data["evecs_list"]

    @abstractmethod
    def get_band_data(self):
        pass


    def get_rsd_data(self, idx_list, path=None):
        pass


    def get_2D_rsd_data(self, idx_list, spin=None):
        pass


