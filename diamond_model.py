from system_model import System
import numpy as np

class Diamond(System):
    # 클래스 변수 -----------------------------------------------------------
    KPOINTS = 1
    MAX_TICKS = 16

    param_names = 'Delta t t_SO lda B n m'.split()
    name = "Diamond"

    # 인스턴스 함수 정의--------------------------------------------------------------------------------------------
    def __init__(self, a, params):
        super().__init__(a, params)
        self.params[5] = int(params[5])
        self.params[6] = int(params[6])
        self.filename = f"diamond_delta={self.params[0]}_t={self.params[1]}_t_SO={self.params[2]}_lda={self.params[3]}_B={self.params[4]}_n={self.params[5]}_m={self.params[6]}.npz"
        self.basis_num = int(2 * self.params[5] * self.params[6])    # spin 고려 안 한 기저 개수
        self.state_list = np.asarray(range(self.basis_num * 2))  # state에 번호 붙인 리스트, spin 고려 o
        self.corner_states = None
    
    def s_decomp(self, i):
        n = self.params[5]
        s1 = 0 if i % (2 * n) < n else 1
        s2 = i % (2 * n) - n * s1
        s3 = i // (2 * n)

        return [s1, s2, s3]

    def s_comp(self,s1, s2, s3):
        n = self.params[5]
        return n * s1 + s2 + 2 * n * s3
    
    def H(self, k):
        delta, t, t_SO, lda, B, n, m = self.params
        a = self.a

        def n_n(i):
            s1, s2, s3 = self.s_decomp(i)
            list = []

            s1_new = s1 + (-1) ** s1
            s2_new = s2 + (-1) ** s1
            s3_new = s3 - (-1) ** s1

            candi = [[s1_new, s2, s3],
                         [s1_new, s2_new, s3],
                         [s1_new, s2, s3_new]]

            for n_n in candi:
                if 0 <= n_n[0] < 2 and 0 <= n_n[1] < n and 0 <= n_n[2] < m:
                    list.append(n_n)

            # print(f"i={i}:{n_n_candi}/{n_n_list}")
            return list

        def next_n_n(i):
            s1, s2, s3 = self.s_decomp(i)
            list = []
            weight = []

            candi = [[s1, s2 - 1, s3 - 1], [s1, s2, s3 - 1],
                              [s1, s2 - 1, s3], [s1, s2 + 1, s3],
                              [s1, s2, s3 + 1], [s1, s2 + 1, s3 + 1]]
            weight_candi = [site * w for w in [1, -1, -1, 1, 1, -1]]
            
            for i, next_n_n in enumerate(candi):
                if 0 <= next_n_n[0] < 2 and 0 <= next_n_n[1] < n and 0 <= next_n_n[2] < m:
                    list.append(next_n_n)
                    weight.append(weight_candi[i])

            return list, weight

        def set_elt(M, i, j, val):
            if i < 0 or j < 0: return
            try:
                M[i, j] = val
            except IndexError:
                pass

        # Hopping term
        H_hop = np.zeros((self.basis_num, self.basis_num), dtype=complex)
        for i in range(self.basis_num):
            n_n_list = n_n(i)
            for j in range(self.basis_num):
                j_decomp = self.s_decomp(j)
                if j_decomp in n_n_list:
                    set_elt(H_hop, i, j, -t)

        # On-site Energy term
        H_on_site = np.zeros((self.basis_num, self.basis_num), dtype=complex)
        for i in range(self.basis_num):
            for j in range(self.basis_num):
                if i == j:
                    if i % (n * 2) < n:
                        set_elt(H_on_site, i, j, delta)
                    else:
                        set_elt(H_on_site, i, j, -delta)

        # SOC
        H_SOC = np.zeros((self.basis_num, self.basis_num), dtype=complex)
        for i in range(self.basis_num):
            site = 1 if i % (n * 2) < n else -1  # a-site랑 b-site 구분
            next_n_n_list, weight = next_n_n(i)

            for j in range(self.basis_num):
                j_decomp = self.s_decomp(j)
                if j_decomp in next_n_n_list:
                    set_elt(H_SOC, i, j, weight[next_n_n_list.index(j_decomp)] * 1j * t_SO)

        I = np.array([[1, 0], [0, 1]])
        S_z = np.array([[1, 0], [0, -1]])

        # Zeeman
        H_zeeman = lda * B * np.eye(self.basis_num)
        S_y = np.array([[0, -1j], [1j, 0]])

        # Hamiltonian
        H = (np.kron(H_hop + H_on_site, I)
              + np.kron(H_SOC, S_z)
              + np.kron(H_zeeman, S_y))
        print(H)
        return H

    # corner state setting method
    def set_corner_states(self):
        self.corner_states = []
        prev_E = self.evals_list[0][0]
        for x in self.state_list[1:]:   # 첫 번째 값은 제외하고 시작
            E = self.evals_list[0][x]
            if prev_E < 0 and E > 0:
                self.corner_states = [x - 1, x]
            prev_E = E

    # y축 범위
    def get_E_bounds(self):
        self.is_E_bounded = True
        E_bounds = [-0.3, 0.3]
        return E_bounds

    def get_band_data(self):

        state_list = self.state_list

        size = 20

        if self.corner_states is None: self.set_corner_states()
        color = (self.basis_num * 2) * ['gray']
        for state in self.corner_states: color[state] = 'blue'

        E_bounds = self.get_E_bounds()

        band_data = state_list, size, color, E_bounds, ""
        return band_data

    def get_path(self, path=0):  # spin 고려 x
        path_sites = []
        path_name = None

        n, m = self.params[5], self.params[6]

        if path == 0:  # diamond 전체
            path_sites = range(2 * n * m)
            path_name = "entire diamond"

        elif path == 1:  # 위쪽 빨간 테두리
            for i in range(n - 1):
                path_sites.append(i)
            for i in range(m):
                path_sites.append(n - 1 + i * 2 * n)
            path_name = "a-site edge"

        elif path == 2:  # 아래쪽 파란 테두리
            b = None
            for i in range(m):
                b = i * 2 * n + n
                path_sites.append(b)
            for i in range(n - 1):
                b += 1
                path_sites.append(b)
            path_name = "b-site edge"

        elif path == 3:  # 가로 지그재그
            for i in range(min(n,m)):
                a = i * (2 * n + 1)
                path_sites.append(a)  # a-site
                if i < min(n,m) - 1:  # a에서 시작해서 a에서 끝나기 때문
                    path_sites.append(a + n + 1)  # b-site
            path_name = "horizontal zigzag"

        elif path == 4:  # 세로
            for i in range(min((n,m))):
                a = n - 1 + i * (2 * n - 1)
                b = a + n
                path_sites.append(a)
                path_sites.append(b)
            path_name = "vertical line"

        return path_sites, path_name

    def get_rsd_data(self, states, path=0):
        site_list, path_name = self.get_path(path=path)  # spin 고려 x
        density_list = []

        color_list = []

        for state in states:
            density = []
            E = self.evals_list[0][state]
            psi = self.evecs_list[0][state]

            # 주요 부분~~~~~~~~~~~~
            for site in site_list:
                density.append(np.abs(psi[2 * site]) ** 2 + np.abs(psi[2 * site + 1]) ** 2)

            if np.isclose(E, 0, atol=1e-1, rtol=1e-1):
                color_list.append('blue')
            else:
                color_list.append('black')

            density_list.append(density)
            # ~~~~~~~~~~~~~~~~~~~~~

        path_site_num = len(site_list)
        xlist = range(path_site_num)

        xlabels = site_list
        step = max(1, path_site_num // self.MAX_TICKS)

        rsd_data = xlist, density_list, states, xlabels, step, path_name, color_list
        return rsd_data

    def get_2D_rsd_data(self, states, spin=None):
        site_list = np.asarray(range(self.basis_num))  # spin 고려 x
        density_list = []

        n, m = self.params[5], self.params[6]

        for site in site_list:
            density = 0
            for state in states:
                psi = self.evecs_list[0][state]
                spin_up = np.abs(psi[2 * site]) ** 2
                spin_down = np.abs(psi[2 * site + 1]) ** 2

                if spin is None:
                    density += spin_up + spin_down
                elif spin == "up":
                    density += spin_up
                elif spin == "down":
                    density += spin_down

            density_list.append(density)

        r3 = np.sqrt(3)

        def get_psi2(x, y):
            y_mod_3 = -1 if y % 3 == 2 else 1

            def div(a, b, de):  # 나눠떨어질 경우에만 몫 반환
                if np.isclose(a % b, 0):
                    return a // b
                else:
                    return de

            s1 = (1 - y_mod_3) // 2
            s2 = div(np.round(r3 * x + y + 2 * s1 - 1), 6, n)
            s3 = div(np.round(r3 * x - 3 * s2), 3, m)
            s = int(np.round(n * s1 + s2 + 2 * n * s3))

            if 0 <= s1 < 2 and 0 <= s2 < n and 0 <= s3 < m:
                # print(f"({x},{y}): s={s}")
                return density_list[s]
            else:
                return 0

        # x축, y축 범위 설정
        x = np.arange(-r3, r3 * (n + m), r3)
        y = []
        for i in range(3 * n + 1):
            y_i = i + 1
            if y_i % 3 != 0:
                y.append(y_i)
                y.append(-y_i)
        y.sort()

        size = 4.5/m * 50
        d_min = 0
        d_max = np.max(density_list)

        data = x, y, get_psi2, size, d_min, d_max
        return data