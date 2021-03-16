import numpy as np
import matplotlib.pyplot as plt
import random

def grid_search_plot(maps_non_adaptive, maps_adaptive):
    # Set the ranges of values tested
    alphas = [3, 5, 7, 9]
    rhos = [0, 0.0005, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5]

    maps_both = np.random.rand(8,4)
    print(maps_both)

    maps_both[0,:] = maps_non_adaptive
    print(maps_both)

    maps_both[1:8,:] = maps_adaptive
    print(maps_both)


    ALPHAS, ROHS = np.meshgrid(alphas, rhos)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(ALPHAS, ROHS, maps_both, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('alpha')
    ax.set_ylabel('rohs')
    ax.set_zlabel('mAP')
    ax.set_title('Grid search: roh vs alpha')
    plt.show()

def main():
    maps_adaptive = np.linspace(0,27,num=28)
    maps_adaptive = np.reshape(maps_adaptive, (7,4), order='F')
    maps_non_adaptive = np.linspace(0,4,num=4)
    grid_search_plot(maps_non_adaptive, maps_adaptive)

if __name__ == "__main__":
    main()