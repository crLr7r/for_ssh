from system_model import System
import numpy as np

class Ribbon(System):
    # 클래스 변수 -----------------------------------------------------------
    KPOINTS = 501
    MAX_TICKS = 16

    param_names = 'Delta t t_SO lda B n'.split()
    name = "Ribbon"

    # 인스턴스 함수 정의--------------------------------------------------------------------------------------------
    def __init__(self, a, params):
        super().__init__(a, params)
        self.params[5] = int(params[5])
        self.filename = f"Ribbon_delta={self.params[0]}_n={self.params[5]}.npz"
        self.basis_num = int(2 * (self.params[5] + 1))  # spin 고려 안 한 기저 개수
        self.kmin, self.kmax = 0, 2 * np.pi / self.a
        self.klist = self.get_klist()

    def H(self, k, k_y=0):
        delta, t, t_SO, lda, B, n = self.params
        a = self.a

        g = -2 * t * np.cos(k * a / 2)
        f1 = -2 * t_SO * np.sin(k * a)
        f2 = 2 * t_SO * np.sin(k * a / 2)

        # Hopping term
        H_hop = np.zeros((self.basis_num, self.basis_num), dtype=complex)
        for i in range(self.basis_num):
            for j in range(self.basis_num):
                if np.abs(i - j) == 1:
                    if (i + j) % 4 == 1:
                        H_hop[i, j] = g
                    elif (i + j) % 4 == 3:
                        H_hop[i, j] = -t

        # On-site Energy term
        H_on_site = np.zeros((self.basis_num, self.basis_num), dtype=complex)
        for i in range(self.basis_num):
            for j in range(self.basis_num):
                if i == j:
                    if i % 2 == 0:
                        H_on_site[i, j] = delta
                    elif i % 2 == 1:
                        H_on_site[i, j] = -delta

        # SOC
        H_SOC = np.zeros((self.basis_num, self.basis_num), dtype=complex)
        weight = 1  # a site과 b site 구분
        for i in range(self.basis_num):
            for j in range(self.basis_num):
                if i == j:
                    H_SOC[i, j] = f1
                if np.abs(i - j) == 2:
                    H_SOC[i, j] = f2
                H_SOC[i, j] *= weight
            weight = -weight

        I = np.array([[1, 0], [0, 1]])
        S_z = np.array([[1, 0], [0, -1]])

        # Zeeman
        H_zeeman = lda * B * np.eye(self.basis_num)
        S_y = np.array([[0, -1j], [1j, 0]])

        # Hamiltonian
        return (np.kron(H_hop + H_on_site, I)
                + np.kron(H_SOC, S_z)
                + np.kron(H_zeeman, S_y))

    # 아래 edge에 가까운지 위 edge에 가까운지 판단
    def is_up_edge(self, evec):
        up_weight = np.sum(np.abs(evec[:4]) ** 2)
        down_weight = np.sum(np.abs(evec[-4:]) ** 2)

        return up_weight > down_weight

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
    def get_edge_band(self, ver=1):
        klist, evals_list, evecs_list = self.klist, self.evals_list, self.evecs_list
        band_idxs = np.zeros(len(klist), dtype=int)
        spin = np.zeros(len(klist), dtype=float)
        edge = np.zeros(len(klist), dtype=int)

        # 기준점 지정
        stand_k_idx = len(klist) // 2  # M 지점에서 시작

        evals = evals_list[stand_k_idx]

        negative_indices = np.where(evals < 0)[0]
        positive_indices = np.where(evals > 0)[0]

        max_idx_n = negative_indices[-2:][::-1]
        min_idx_p = positive_indices[:2]

        idxs = np.concatenate((max_idx_n, min_idx_p))  # = max_idx_n 과 min_idx_p를 이어붙임

        stand_idx = idxs[ver - 1]

        band_idxs[stand_k_idx] = stand_idx
        spin[stand_k_idx] = self.get_avg_spin(evecs_list[stand_k_idx][stand_idx])
        if self.is_up_edge(evecs_list[stand_k_idx][stand_idx]): edge[stand_k_idx] = 1

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

            band_idxs[i] = this_idx
            spin[i] = self.get_avg_spin(evecs_list[i][this_idx])
            if self.is_up_edge(evecs_list[i][this_idx]): edge[i] = 1

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

            band_idxs[i] = this_idx
            spin[i] = self.get_avg_spin(evecs_list[i][this_idx])
            if self.is_up_edge(evecs_list[i][this_idx]): edge[i] = 1

            prev_k_idx = i
            prev_idx = this_idx

        return band_idxs, spin, edge

    def set_kspace_data(self):
        self.klabel = [[self.kmin, self.kmax/2, self.kmax]
                     , [r"$0$", r"$\pi/a$", r"$2\pi/a$"]]
        kpw = self.kmax - self.kmin
        self.line_indicator = [[kpw/3, kpw/2, 2*kpw/3],[]]
        self.spec_indices = None

    # y축 범위
    def get_E_bounds(self):
        self.is_E_bounded = True
        E_bounds = [-1, 1]
        return E_bounds

    # DataPlot에서 draw_band(model, data)할 때 반환 값이 data로 들어감
    def get_band_data(self):

        state_list = self.klist

        size = 0.5

        color = 'black'

        E_bounds = self.get_E_bounds()

        band_data = state_list, size, color, E_bounds, ""
        return band_data

    #DataPlot에서 draw_kspace_val(model, data)에서 반환값이 data로 들어감
    def get_spin_data(self, ver=1):
        band_idxs, _ , _ = self.get_edge_band(ver=ver)
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
    def get_HL_band_data(self, ver_list, distinct="spin"):
        klist = self.klist
        size = 0.5

        results = [self.get_edge_band(ver=i) for i in ver_list]
        bands = [r[0] for r in results]
        spins = [r[1] for r in results]
        edges = [r[2] for r in results]

        Elist_list = self.evals_list.T
        colors_list = []

        # colors_list 설정 ~~~~~~~~~~
        for i in range(len(Elist_list)):
            colors = ['black'] * len(klist)
            for j in range(len(klist)):
                for g in range(len(bands)):
                    if i == bands[g][j]:
                        if distinct == "spin":
                            colors[j] = self.get_spin_color(spins[g][j])
                        elif distinct == "edge":
                            if edges[g][j] == 0:
                                colors[j] = 'blue'
                            else:
                                colors[j] = 'red'
                        else:
                            colors[j] = 'darkturquoise'

            colors_list.append(colors)

        # ~~~~~~~~~~~~~~~~~~~~

        E_bounds = self.get_E_bounds()

        if distinct == "spin":
            labels = ["spin up = red", "spin down = blue"]
        elif distinct == "edge":
            labels = ["upper edge = red", "lower edge = blue"]
        else:
            labels = ["edge state"]

        textstr = "\n".join(labels)

        band_data = klist, size, colors_list, E_bounds, textstr
        return band_data

    def get_xlist(self, path="basis"):
        xlist = None
        if path == "basis":
            xlist = range(self.basis_num)

        elif path == "y":
            xlist = np.zeros(self.basis_num)
            for i in range(self.basis_num - 1):
                if i % 2 == 0:
                    xlist[i + 1] = xlist[i] + self.a / (2 * np.sqrt(3))
                else:
                    xlist[i + 1] = xlist[i] + self.a / np.sqrt(3)

        elif path == "k":
            xlist = self.klist

        return np.asarray(xlist)

    def set_xi_data(self, ver=1, log=True):
        xi_list = []
        band_idxs, _, _ = self.get_edge_band(ver=ver)

        for i in range(self.KPOINTS):
            x = self.get_xlist()
            density = np.zeros(self.basis_num * 2)
            evals, evecs = self.evals_list[i], self.evecs_list[i]
            idx = band_idxs[i]
            psi = evecs[idx]

            for j in range(self.basis_num * 2):
                density[j] += np.abs(psi[j]) ** 2

            density = density[0::2] + density[1::2]

            x_a, x_b = x[0::2], x[1::2]
            density_a, density_b = density[0::2], density[1::2]

            if np.sum(density_a) > np.sum(density_b):
                x = x_a
                density = density_a
            else:
                x = x_b
                density = density_b

            _, _, xi, _ = self.get_exp_fit(x, density, log=log)
            xi_list.append(xi)

        self.xi_data = xi_list, "xi", 0.5

    def get_rsd_data(self, states, path="basis"):

        basis = self.basis_num * 2  # spin을 포함한 basis 개수
        band_idxs, _, _ = self.get_edge_band(ver=1)

        x = self.get_xlist(path=path)

        xi_list, _, _ = self.xi_data
        max_idx = [np.argmax(xi_list[:self.KPOINTS // 3]), np.argmax(xi_list[2 * self.KPOINTS // 3:]) + 2 * self.KPOINTS // 3]

        xlabels = []
        if path=="basis":
            for i in range(self.basis_num // 2):
                xlabels.append(fr"$a_{{{i}}}$")
                xlabels.append(fr"$b_{{{i}}}$")
        else:
            xlabels = x

        density_list = []
        color_list = []

        for i in range(len(states)):
            # if i == boundary[1]: continue  # 마지막 제외
            density = np.zeros(basis)

            evals, evecs = self.evals_list[i], self.evecs_list[i]
            idx = band_idxs[i]
            psi = evecs[idx]

            for j in range(basis):
                density[j] += np.abs(psi[j]) ** 2

            density = density[0::2] + density[1::2] # spin 자유도 제거

            if path == "basis":
                density_a, density_b = density[0::2], density[1::2]
                if np.sum(density_a) > np.sum(density_b):
                    density = density_a
                else:
                    density = density_b

            density_list.append(density)
            color_list.append('red' if (i in max_idx) else 'black')

        if path == "basis":
            x_a, x_b = x[0::2], x[1::2]
            density_a, density_b = density_list[0][0::2], density_list[0][1::2]
            xlabels_a, xlabels_b = xlabels[0::2], xlabels[1::2]

            if np.sum(density_a) > np.sum(density_b):
                x = x_a
                xlabels = xlabels_a
            else:
                x = x_b
                xlabels = xlabels_b

        step = len(x) // min(self.MAX_TICKS, len(x))

        rsd_data = x, density_list, states, xlabels, step, path, color_list
        return rsd_data