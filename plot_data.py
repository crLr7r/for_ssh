from system_plot import DataPlot
from diamond_model import Diamond

import sys

if __name__ == '__main__':
    model_name, a, params = (sys.argv[1],
                                 float(sys.argv[2]),
                                 [float(x) for x in sys.argv[3:]])

    model = None
    states = None
    is_kspace = None
    if model_name == 'Diamond':
        model = Diamond(a, params)
        is_kspace = False
        model.load_eigen_data()
        model.set_corner_states()
        states = model.corner_states  # 가변적



    # 가변적 ---------------------------------------------------------------
    DataPlot.draw_band(model, model.get_band_data(), is_E_bounded=True, is_x_bounded=True, is_kspace=is_kspace)
    DataPlot.draw_rsd(model, model.get_rsd_data(states, path=1), log=True)
    DataPlot.draw_2D_rsd(model, model.get_2D_rsd_data(states))
