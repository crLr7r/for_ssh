import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class DataPlot():
    @staticmethod
    # kspace에서 그래프 그리는 기반
    def set_kspace_diagram(model, ax=None, title=None, fig_size=(6.4, 4.8)):
        model.set_kspace_data()
        model.set_title(title=title)

        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False

        for i in model.line_indicator[0]:
            ax.axvline(i, color="gray", linewidth=0.8, alpha=0.4)

        for i in model.line_indicator[1]:
            ax.axhline(i, color="gray", linewidth=0.8, alpha=0.4)

        if model.spec_indices is not None:
            for i in model.spec_indices[0]:       # x축
                ax.axvline(i, color="red", linewidth=0.8, alpha=0.8)
            for i in model.spec_indices[1]:       # y축
                ax.axhline(i, color="red", linewidth=0.8, alpha=0.8)

        if created_fig:
            ax.set_title(model.title)
            ax.set_xticks(model.klabel[0], model.klabel[1])
            ax.set_xlim(model.kmin, model.kmax)
            ax.set_xlabel("k")

        return fig, ax, created_fig

    @staticmethod
    def draw_band(model, band_data,
                  ax=None, title=None, fig_size=(6.4, 4.8),
                  is_kspace=True, is_E_bounded=True, is_x_bounded=False, is_last=True):

        state_list, size, color, E_bounds = band_data

        if ax is None:
            if is_kspace:
                fig, ax, created_fig = DataPlot.set_kspace_diagram(model, ax=ax, title=title, fig_size=fig_size)
            else:
                fig, ax = plt.subplots(figsize=fig_size)
                model.set_title(title=title)
        else:
            fig = ax.figure

        if is_E_bounded:
            ax.set_ylim(E_bounds[0], E_bounds[1])

        if len(model.evals_list) > 1:
            evals_list = model.evals_list.T
            for Elist in evals_list:
                ax.scatter(state_list, Elist, s=size, color=color)
        else:
            ax.scatter(state_list, model.evals_list[0], s=size, color=color)
            if is_E_bounded and is_x_bounded:
                x_min, x_max = (state_list[DataPlot.find_closest_idx(model.evals_list[0], E_bounds[0])],
                                state_list[DataPlot.find_closest_idx(model.evals_list[0], E_bounds[1], smaller=False)])
                ax.set_xlim(x_min, x_max)

        if is_last:
            ax.set_ylabel('E(k)') if is_kspace else ax.set_ylabel('E')
            if not is_kspace: ax.set_xlabel('state')
            fig.tight_layout()
            fig.savefig(f"{model.name}_band({model.title}).png")
            plt.show()
            plt.close(fig)

    @staticmethod
    def find_closest_idx(lst, target, smaller=True):
        # target보다 작은 (값, 인덱스) 쌍만 필터링
        if smaller:
            candidates = [(val, idx) for idx, val in enumerate(lst) if val < target]
        else:
            candidates = [(val, idx) for idx, val in enumerate(lst) if val > target]

        # 조건에 맞는 값이 없으면 None 반환
        if not candidates:
            return None

        # 값이 가장 큰(target에 가장 가까운) 요소의 인덱스 반환
        if smaller:
            return max(candidates, key=lambda x: x[0])[1]
        else:
            return min(candidates, key=lambda x: x[0])[1]

    # 주어진 y값들에 대해 극대 인덱스 리스트 반환
    @staticmethod
    def get_peak(ylist):
        peak_idx_list = []
        is_increase = True
        prev_y = ylist[0]
        for i, y in enumerate(ylist[1::]):
            if prev_y - y > 0.01:  # 의미있는 감소를 했는데
                if is_increase: peak_idx_list.append(i)  # 이전까지 증가했을 때(i-1이 아니라 i임 주의)
                is_increase = False
            elif prev_y - y < -0.01:  # 의미있는 증가만 증가로 취급
                is_increase = True
            else:
                is_increase = False
            prev_y = y
        if is_increase: peak_idx_list.append(len(ylist) - 1)  # 마지막까지 증가했으면 마지막 인덱스도 추가

        return peak_idx_list

    @staticmethod
    def draw_rsd(model, rsd_data,
                 ax=None, title=None,
                 fig_size=(10, 5), is_line=True, log=False, is_last=True):

        #idx_list: 그래프를 여러개 그린다고 했을 때 그래프끼리 구분해주는 list
        xlist, density_list, states, xlabels, step, path, color_list = rsd_data

        model.set_title(title=title)

        if ax is None: fig, ax = plt.subplots(figsize=fig_size)
        else: fig = ax.figure

        for i, density in enumerate(density_list):

            if log:
                density = np.log(np.maximum(density, 1e-13))

            if is_line:
                ax.plot(xlist,density,marker="o",markersize=3,linewidth=1, label=f"state={states[i]}", color=color_list[i])
            else:
                ax.plot(xlist,density,marker="o",markersize=3,linestyle="none",label=f"state={states[i]}", color=color_list[i])

            peak_idxs = DataPlot.get_peak(density)
            for j in peak_idxs:
                ax.annotate(
                    f"({xlabels[j]},{density[j]:1.3f})",
                    xy=(xlist[j], density[j]),  # 가리킬 점의 실제 좌표
                    xytext=(xlist[j] + 3, density[j] - 0.03),  # 텍스트가 나타날 좌표
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5)  # 화살표 옵션
                )

        if is_last:
            ax.legend()
            ax.set_title(model.title)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlabel(path)
            ax.set_ylabel(r"$|\psi|^2$") if not log else ax.set_ylabel(r"ln($|\psi|^2$)")
            xticks = xlist[::step], xlabels[::step]
            ax.set_xticks(*xticks)

            fig.tight_layout()
            fig.savefig(f"{model.name}_{path}_rsd_k({model.title}).png")

            plt.show()
            plt.close()

    @staticmethod
    def draw_2D_rsd(model, rsd_2D_data,
                    ax=None, title=None):
        x, y, d_func, size, d_min, d_max = rsd_2D_data

        # pcolormesh 그리기
        model.set_title(title=title)
        X, Y = np.meshgrid(x, y)

        D = np.zeros_like(X, dtype=float)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                D[i, j] = d_func(X[i, j], Y[i, j])

        D_masked = np.ma.masked_where(D == 0, D)

        if ax is None:
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)

            height = 8
            width = height * x_range / y_range

            fig, ax = plt.subplots(figsize=(width, height))

        else:
            fig = ax.figure

        # Blues의 0.2 ~ 1.0 부분만 사용
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'truncated_blues',
            plt.get_cmap('Blues')(np.linspace(0.3, 1.0, 256))
        )
        sc = ax.scatter(
            X.ravel(),
            Y.ravel(),
            c=D_masked.ravel(),
            s=size,
            cmap=new_cmap,
            vmin=d_min,
            vmax=d_max
        )

        fig.colorbar(sc, ax=ax, label=r"$|\psi|^2$")
        ax.set_aspect('auto')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(model.title)
        fig.savefig(f"{model.name}_rsd_2D({model.title}).png")
        plt.show()