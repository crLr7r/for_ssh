import os
from save_data import save_eigen_data
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy.optimize import curve_fit
from PATH import DATA_DIR

# 실행용 파일 아님

class System(ABC):
    # 클래스 변수 -----------------------------------------------------------
    KPOINTS = 501
    MAX_TICKS = 16

    param_names = None
    name = None

    # 인스턴스 함수 정의--------------------------------------------------------
    @abstractmethod
    def __init__(self, a, params):  # params과 param_names는 일대일 대응

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

    def load_eigen_data(self, type="npz"):
        if type == "npz":
            path = f"{DATA_DIR}/{self.filename}.npz"
            if not os.path.exists(path):
                print(f"file does not exist; saving data first...({self.filename})")
                save_eigen_data(self)

            with np.load(path) as data:
                self.evals_list = data["evals_list"]
                self.evecs_list = data["evecs_list"]

        elif type == "txt":
            pass

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

    # 주어진 y값들에 대해 극대 인덱스 리스트 반환
    @staticmethod
    def get_peak(ylist):
        ylist_len = len(ylist)
        yscale = max(ylist) - min(ylist)
        
        def idx(i):
            if i < 0: return 0
            elif i >= ylist_len: return ylist_len - 1
            else: return i
        
        def is_decrease(i, j):
            i, j = idx(i), idx(j)
            if i == j: return 1     # j가 실제 인덱스가 아닐 때
            if ylist[i] - ylist[j] > yscale / 50: return 1 # 의미 있는 감소를 했을 때
            elif ylist[i] > ylist[j]: return 0  # 작지만 감소는 했을 때
            else: return -1 # 감소도 안 했을 때

        def is_peak(i): # 앞 뒤 3개를 보고 peak인지 아닌지 판단한다
            
            m1, m2, m3 = is_decrease(i, i-1), is_decrease(i, i-2), is_decrease(i, i-3)
            p1, p2, p3 = is_decrease(i, i+1), is_decrease(i, i+2), is_decrease(i, i+3)

            is_left_decrease = 4*m1 + 2*m2 + m3 > 0
            '''
            if m1 == 1: is_left_decrease = True
            elif m1 == 0:
                if m2 == 1: is_left_decrease = True
                elif m2 == 0:
                    if m3 == 1: is_left_decrease = True
            '''
            is_right_decrease = 4*p1 + 2*p2 + p3 > 0
            '''
            if p1 == 1: is_right_decrease = True
            elif p1 == 0:
                if p2 == 1: is_right_decrease = True
                elif p2 == 0:
                    if p3 == 1: is_right_decrease = True
            '''

            return is_left_decrease and is_right_decrease
        
        peak_idx_list = []
        
        for i in range(ylist_len):
            if is_peak(i): peak_idx_list.append(i)

        '''
        is_increase = True
        prev_y = ylist[0]
        for i, y in enumerate(ylist[1::]):
            if prev_y - y > 0.001:  # 의미있는 감소를 했는데
                if is_increase: peak_idx_list.append(i)  # 이전까지 증가했을 때(i-1이 아니라 i임 주의)
                is_increase = False
            elif prev_y - y < -0.001:  # 의미있는 증가만 증가로 취급
                is_increase = True
            else:
                is_increase = False
            prev_y = y
        if is_increase: peak_idx_list.append(len(ylist) - 1)  # 마지막까지 증가했으면 마지막 인덱스도 추가
        '''
        return peak_idx_list

    @staticmethod
    # 피팅할 함수: 지수 감쇠 함수
    def exp_dissolve(x, A, B, xi, C):
        return A * np.exp(-np.abs(x - B) / xi) + C
    @staticmethod
    # 피팅할 함수2: 세미로그 함수
    def exp_log_dissolve(x, A, B, xi, C):
        return np.log(A) - np.abs(x - B) / xi

    @staticmethod
    # 어떤 xlist, dlist 데이터에 대해 지수 감쇠 모델을 피팅하는 함수
    def get_exp_dissolve_fit(xlist, dlist, log=True, return_log=True):  # xi_0: 초기 추정값
        x = np.asarray(xlist)
        y = np.asarray(dlist)

        #if y[-1] > y[0]: y = y[::-1]  # 반대 그래프의 경우 순서 뒤집기

        #print(x)
        # 초기 추정값
        C0 = np.min(y)
        B0 = x[np.argmax(y)]
        xi0 = np.abs(x[-1] - x[0]) / 5
        A0 = np.max(y) - C0
        #print(f"초기값: {A0}, {xi0}, {B0}, {C0}")
        
        # 로그 스케일일 경우
        fit_func = System.exp_dissolve
        if log:
            y = np.log(np.maximum(y, 1e-13))
            fit_func = System.exp_log_dissolve
            '''
            for idx in range(len(y)):
                if idx == 0: continue
                if np.isclose(y[idx], y[idx - 1], rtol=1e-8, atol=1e-10):
                    y = y[:idx]
                    x = x[:idx]
                    break
            '''

            peak_idx_list = System.get_peak(y)

            if len(peak_idx_list) > 0:
                peak_idx = peak_idx_list[0]
            
                y = y[peak_idx:]
                x = x[peak_idx:]

        # 피팅
        parameters, covariance = curve_fit(
            fit_func,
            x,
            y,
            p0=[A0, B0, xi0, C0],  # A, B, xi, C의 초기 추정값
            bounds=([0, 0, 1e-10, 0], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=100000
        )
        if log: parameters[3] = 0
        if return_log: fit_func = System.exp_log_dissolve
        else: fit_func = System.exp_dissolve
        x_fit = x
        y_fit = fit_func(x_fit, *parameters)

        parameter_errors = np.sqrt(np.diag(covariance))
        
        return parameters, parameter_errors, x_fit, y_fit

    @staticmethod
    # 피팅할 함수: 지수 함수
    def exp(x, A, xi, C):
        return A * np.exp(x / xi) + C

    @staticmethod
    # 피팅할 함수2: 세미로그 함수
    def exp_log(x, A, xi, C):
        return np.log(A) + x / xi

    @staticmethod
    def get_exp_fit(xlist, dlist, log=True, return_log=True):  # xi_0: 초기 추정값
        x = np.asarray(xlist)
        y = np.asarray(dlist)

        #if y[-1] > y[0]: y = y[::-1]  # 반대 그래프의 경우 순서 뒤집기
        #print(x)
        # 초기 추정값
        C0 = np.min(y)
        xi0 = np.abs(x[-1] - x[0]) / 5
        A0 = np.max(y) - C0
        #print(f"초기값: {A0}, {xi0}, {B0}, {C0}")
        
        # 로그 스케일일 경우
        fit_func = System.exp
        if log:
            y = np.log(np.maximum(y, 1e-13))
            fit_func = System.exp_log
            '''
            for idx in range(len(y)):
                if idx == 0: continue
                if np.isclose(y[idx], y[idx - 1], rtol=1e-8, atol=1e-10):
                    y = y[:idx]
                    x = x[:idx]
                    break
            '''
            '''
            peak_idx_list = System.get_peak(y)
            peak_idx = peak_idx_list[0]
            
            y = y[peak_idx + 2:]
            x = x[peak_idx + 2:]
            '''
        # 피팅
        parameters, covariance = curve_fit(
            fit_func,
            x,
            y,
            p0=[A0, xi0, C0],  # 초기 추정값
            maxfev=100000
        )

      
        parameter_errors = np.sqrt(np.diag(covariance))
        
        if log: parameters[2] = 0   # C=0
        if return_log: fit_func = System.exp_log
        else: fit_func = System.exp
        x_fit = x
        y_fit = fit_func(x_fit, *parameters)
        
        return parameters, parameter_errors, x_fit, y_fit

    # 피팅할 함수: power 함수
    @staticmethod
    def power(x, A, alpha):
        return A * (x ** alpha)

    @staticmethod
    def power_log(x, A, alpha):
        return np.log2(A) + alpha * np.log2(x)

    @staticmethod
    def get_power_fit(xlist, dlist, log=True, return_log=False):  # xi_0: 초기 추정값
        x = np.asarray(xlist)
        y = np.asarray(dlist)

        #if y[-1] > y[0]: y = y[::-1]  # 반대 그래프의 경우 순서 뒤집기
        #print(x)
        
        # 로그 스케일일 경우
        fit_func = System.power
        if log: 
            fit_func = System.power_log
            y = np.log(y)
        '''
        for idx in range(len(y)):
            if idx == 0: continue
            if np.isclose(y[idx], y[idx - 1], rtol=1e-8, atol=1e-10):
                y = y[:idx]
                x = x[:idx]
                break
        '''
        '''
        peak_idx_list = System.get_peak(y)
        peak_idx = peak_idx_list[0]
            
        y = y[peak_idx + 2:]
        x = x[peak_idx + 2:]
        '''
        # 피팅
        parameters, covariance = curve_fit(
            fit_func,
            x,
            y,
            p0=[1, -1],
            maxfev=100000
        )

        if return_log: fit_func = System.power_log
        else: fit_func = System.power

        x_fit = x
        y_fit = fit_func(x_fit, *parameters)

        parameter_errors = np.sqrt(np.diag(covariance))
        
        return parameters, parameter_errors, x_fit, y_fit