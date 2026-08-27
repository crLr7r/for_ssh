from system_model import System
import numpy as np

#실행용 파일 아님
class Bulk(System):
    # 클래스 변수 -----------------------------------------------------------
    KPOINTS = 501
    MAX_TICKS = 16

    param_names = 'Delta t t_SO'.split()
    name = "Bulk"

    # 인스턴스 함수 정의--------------------------------------------------------------------------------------------
    def __init__(self, a, params, k_y=0):
        super().__init__(a, params)
        self.k_y = k_y

        self.filename = f"bulk_delta={self.params[0]}_t={self.params[1]}_t_SO={self.params[2]}_k_y={self.k_y}.npz"
        self.basis_num = 2    # spin 고려 안 한 기저 개수
        self.kmin, self.kmax = 0, 4 * np.pi / self.a
        self.klist = self.get_klist()
    
    def H(self, k):
        Delta, t, t_SO = self.params
        a, k_y = self.a, self.k_y

        f_SO = -2 * t_SO * (np.sin(k * a) - 2 * np.cos(k_y*np.sqrt(3)*a/2)*np.sin(k * a/2))
        f_x = -t * (1 + 2 * np.cos(k_y * np.sqrt(3) * a/2) * np.cos(k * a / 2))
        f_y =  -2 * t * np.sin(k_y * np.sqrt(3) * a/2) * np.cos(k * a / 2)
        g = f_x + 1j * f_y

        H = np.array([[f_SO+Delta, 0, g, 0],
                             [0, -f_SO+Delta, 0, g],
                             [g, 0, -f_SO-Delta, 0],
                             [0, g, 0, f_SO-Delta]])
        #print(H)
        return H

    # kspace data 설정
    def set_kspace_data(self):
        kpw = self.kmax - self.kmin
        self.klabel = [[0, kpw/3, kpw/2, 2*kpw/3, kpw]
                      ,[r"$\Gamma$", r"$K$", r"$M$", r"$K'$", r"$\Gamma$"]]
        self.line_indicator = [[kpw/3, kpw/2, 2*kpw/3],[0]]
        self.spec_indices = None

    # y축 범위
    def get_E_bounds(self):
        self.is_E_bounded = True
        E_bounds = [-1, 1]
        return E_bounds

    # evec의 평균 spin z성분 반환
    def get_avg_spin(self, evec):
        spin = 1
        spin_sum = 0
        prob_sum = 0
        for elt in evec:
            prob = np.abs(elt) ** 2
            spin_sum += prob * spin
            prob_sum += prob
            spin = -spin
        return spin_sum / prob_sum

    # spin값에 대응하는 색을 선형 보간으로 구함
    def get_spin_color(self, spin):
        red = np.array([1.0, 0.0, 0.0])
        purple = np.array([0.5, 0.0, 0.5])
        blue = np.array([0.0, 0.0, 1.0])

        if spin >= 0: color = purple * (1 - spin) + red * spin
        else: color = purple * (1 + spin) + blue * (-spin)

        return color
    
    # edge band 고르는 함수
    def get_bands(self):
        klist, evals_list, evecs_list = self.klist, self.evals_list, self.evecs_list
        band_idxs = np.zeros((self.basis_num * 2, len(klist)), dtype=int)
        spin = np.zeros((self.basis_num * 2,len(klist)), dtype=float)

        # 기준점 지정
        stand_k_idx = len(klist) // 2 - 1  # M보다 하나 왼쪽에서 시작
        idxs = range(self.basis_num * 2)
        
        for band_num in idxs:
            stand_idx = band_num
            
            band_idxs[band_num][stand_k_idx] = stand_idx
            spin[band_num][stand_k_idx] = self.get_avg_spin(evecs_list[stand_k_idx][stand_idx])
            
            # 기준점 왼쪽으로
            prev_k_idx = stand_k_idx  # k_i-1
            prev_idx = stand_idx
            for i in range(stand_k_idx - 1, -1, -1):
                prev_evec = evecs_list[prev_k_idx][prev_idx]
                max_dot = 0
                this_idx = 0

                for j in range(self.basis_num * 2):
                    this_evec = evecs_list[i][j]
                    dot = np.abs(np.vdot(prev_evec, this_evec)) ** 2
                    if dot > max_dot:
                        max_dot = dot
                        this_idx = j

                band_idxs[band_num][i] = this_idx
                spin[band_num][i] = self.get_avg_spin(evecs_list[i][this_idx])

                prev_k_idx = i
                prev_idx = this_idx

            # 기준점 오른쪽으로
            prev_k_idx = stand_k_idx  # k_i-1
            prev_idx = stand_idx
            for i in range(stand_k_idx + 1, len(klist), 1):
                prev_evec = evecs_list[prev_k_idx][prev_idx]
                max_dot = 0
                this_idx = 0

                for j in range(self.basis_num * 2):
                    this_evec = evecs_list[i][j]
                    dot = np.abs(np.vdot(prev_evec, this_evec)) ** 2
                    if dot > max_dot:
                        max_dot = dot
                        this_idx = j

                band_idxs[band_num][i] = this_idx
                spin[band_num][i] = self.get_avg_spin(evecs_list[i][this_idx])

                prev_k_idx = i
                prev_idx = this_idx

        return band_idxs, spin

    def get_band_data(self):

        state_list = self.klist

        size = 0.5
        color = 'black'

        E_bounds = self.get_E_bounds()

        band_data = state_list, size, color, E_bounds, ""
        return band_data

    #DataPlot에서 draw_kspace_val(model, data)에서 반환값이 data로 들어감
    def get_spin_data(self, band_num=0):
        band_idxs, _ = self.get_bands()
        band_idxs = band_idxs[band_num]  # 선택한 밴드의 인덱스만 가져오기

        spin_avg_list = []

        for k_idx in range(len(self.klist)):
            evecs, idx = self.evecs_list[k_idx], band_idxs[k_idx]
            evec = evecs[idx]
            spin_avg_list.append(self.get_avg_spin(evec))

        y_name = "avg_spin"
        size = 0.5

        val_data = spin_avg_list, y_name, size
        return val_data

    #DataPlot에서 draw_band(model, data)할 때 반환 값이 data로 들어감
    def get_HL_band_data(self):
        klist = self.klist
        size = 0.5

        bands, spins = self.get_bands()

        Elist_list = self.evals_list.T
        colors_list = []

        # colors_list 설정 ~~~~~~~~~~
        for i in range(len(Elist_list)):
            colors = ['black'] * len(klist)
            for j in range(len(klist)):
                for g in range(len(bands)):
                    if i == bands[g][j]:
                        colors[j] = self.get_spin_color(spins[g][j])

            colors_list.append(colors)

        # ~~~~~~~~~~~~~~~~~~~~

        E_bounds = self.get_E_bounds()

        labels = ["spin up = red", "spin down = blue"]
        textstr = "\n".join(labels)

        band_data = klist, size, colors_list, E_bounds, textstr
        return band_data
