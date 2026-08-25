import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

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
                  is_kspace=True, is_E_bounded=True, is_x_bounded=False, is_text=False, is_last=True):

        state_list, size, color, E_bounds, textstr = band_data

        if ax is None:
            if is_kspace:
                fig, ax, created_fig = DataPlot.set_kspace_diagram(model, ax=ax, title=title, fig_size=fig_size)
            else:
                fig, ax = plt.subplots(figsize=fig_size)
                model.set_title(title=title)
        else:
            fig = ax.figure
            model.set_title(title=title)

        if is_E_bounded:
            ax.set_ylim(E_bounds[0], E_bounds[1])

        if len(model.evals_list) > 1:
            evals_list = model.evals_list.T
            for i, Elist in enumerate(evals_list):
                ax.scatter(state_list, Elist, s=size, color=color[i])
        else:
            ax.scatter(state_list, model.evals_list[0], s=size, color=color)
            if is_E_bounded and is_x_bounded:
                x_min, x_max = (state_list[DataPlot.find_closest_idx(model.evals_list[0], E_bounds[0])],
                                state_list[DataPlot.find_closest_idx(model.evals_list[0], E_bounds[1], smaller=False)])
                ax.set_xlim(x_min, x_max)

        if is_text:
            ax.text(0.05, 0.55, textstr,
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

        if is_last:
            ax.set_ylabel('E(k)') if is_kspace else ax.set_ylabel('E')
            if not is_kspace: ax.set_xlabel('state')
            fig.tight_layout()
            fig.savefig(f"images/{model.name}_band({model.title}).png")
            plt.show()
            plt.close(fig)
        
        return ax

    @staticmethod
    def draw_kspace_val(model, val_data,
                        ax = None, title = None, fig_size = (6.4, 4.8), is_last = True):
        y_list, y_name, size = val_data

        if ax is None:
            fig, ax, created_fig = DataPlot.set_kspace_diagram(model, ax=ax, title=title, fig_size=fig_size)
        else:
            fig = ax.figure

        ax.scatter(model.klist, y_list, s=size, label=y_name)

        if is_last:
            ax.set_ylabel(y_name)
            fig.tight_layout()
            fig.savefig(f"images/{model.name}_{y_name}({model.title}).png")
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

    @staticmethod
    def draw_rsd(model, rsd_data,
                 ax=None, title=None,
                 fig_size=(10, 5), is_line=True, log=False, is_last=True):

        #states: 그래프를 여러개 그린다고 했을 때 그래프끼리 구분해주는 list
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

            peak_idxs = model.get_peak(density)
            #print("peak_idxs=",peak_idxs)
            y_scale = np.max(density) - np.min(density)
            y_mid = (np.max(density) + np.min(density)) / 2
            print(y_scale)
            for j in peak_idxs:
                y_weight = 1 if density[j] > y_mid else -1
                ax.annotate(
                    f"({xlabels[j]},{density[j]:1.3f})",
                    xy=(xlist[j], density[j]),  # 가리킬 점의 실제 좌표
                    xytext=(xlist[j], density[j] - y_weight * y_scale/6),  # 텍스트가 나타날 좌표
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
            fig.savefig(f"images/{model.name}_{path}_rsd_k({model.title}).png")

            plt.show()
            plt.close()

    @staticmethod
    def draw_2D_rsd(model, rsd_2D_data,
                    ax=None, title=None):
        x, y, d_func, size, d_min, d_max = rsd_2D_data

        # pcolormesh 그리기
        model.set_title()
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
        ax.set_title(model.title if title is None else title)
        #fig.tight_layout()
        fig.savefig(f"images/{model.name}_rsd_2D({model.title})(state#1800).png")
        plt.show()

    #임의의 데이터 그리는 용도
    @staticmethod
    def draw_any_data(model, data,
                             ax=None, title="data", is_last=True, 
                             xlog=False, ylog=False, base=2, x_inverse=False, error_bar=False, y_error=None):
        
        xlist, ylist, xlabel, ylabel, label, extra_data = data
        model.set_title(title=title)
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        xtick_labels = xlist
        
        if x_inverse: 
            xtick_labels = []
            for x in xlist: xtick_labels.append(f"1/{round(x,0)}")
            xlist = 1/np.asarray(xlist)
        
        if error_bar: ax.errorbar(xlist, ylist, yerr=y_error, fmt='o-', markersize=3, linewidth=1, color="black", label=label, capsize=3)
        else: ax.plot(xlist, ylist, marker="o", markersize=3, linewidth=1, color="black", label=label)

        xticks = [xlist[i] for i in np.linspace(0, len(xlist) - 1, min(len(xlist), 9), dtype=int)]
        xtick_labels = [xtick_labels[i] for i in np.linspace(0, len(xtick_labels) - 1, min(len(xtick_labels), 9), dtype=int)]
        
        if extra_data is not None:
            x0 = extra_data[0]
            y0 = extra_data[1]
            ax.scatter(x0, y0, marker='x', s=50, color="red")
            yscale = max(ylist) - min(ylist)
            ax.annotate(
                    f"ribbon band gap",
                    xy=(x0, y0),  # 가리킬 점의 실제 좌표
                    xytext=(x0 + 0.001, y0 + yscale/3),  # 텍스트가 나타날 좌표
                    arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.05, width=0.5, headwidth=5),  # 화살표 옵션
                    color="red"
                )
            xticks = np.insert(xticks, 0, x0)
            xtick_labels = np.insert(xtick_labels, 0, x0)
        
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
      
        ax.legend()

        if is_last:
            ax.grid(axis="y", alpha=0.25)
            if x_inverse:
                ax.set_xlabel(f"{xlabel}(1/n)", size=15)
            else:
                ax.set_xlabel(xlabel, size=15)

            ax.set_ylabel(ylabel, size=15)
            
            if xlog: ax.set_xscale('log', base=base)
            if ylog: ax.set_yscale('log', base=base)
            
            subtitle=""
            if xlog and ylog: subtitle = "(log-log)"
            elif xlog and not ylog: subtitle = "(xlog)"
            elif ylog and not xlog: subtitle = "(ylog)"
            
            ax.set_title(f"{model.title}{subtitle}")
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3e'))
            
            fig.tight_layout()
            fig.savefig(f"images/{model.name}_{title}{subtitle} - {xlabel}.png")
            
            plt.show()
            plt.close()