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



def grid_search_plot2():
    results = []   

    x_axis = []
    block_size = [4,8,16,32,64]
    search_border = [2,4,8,16,32]

    metric = [] # Search border x block_size

    pepn = []
    msen = []

    for res in results:
        pepn.append(res[7])
        msen.append(res[6])
        if res[0] == 'forward':
            if res[4] == "SAD":
                # block_size.append(res[1])
                # search_border.append(res[2])
                #metric.append(res[7]) # PEPN
                metric.append(res[6]) # MSEN
    
    print(pepn)
    print(msen)

    pepn.sort()
    msen.sort()

    print("\n")
    print(pepn)
    print(msen)

    BLOCK, SEARCH = np.meshgrid(block_size, search_border)
    metric = np.reshape(metric, (5, 5))

    

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(BLOCK, SEARCH, metric, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('Block size')
    ax.set_ylabel('Search border')
    ax.set_zlabel('MSEN')
    ax.set_title('Forward (SSD)')
    plt.show()

def grid_search_plot3():

    results = []


    x_axis = []
    block_size = [4,8,16,32,64]
    search_border = [2,4,8,16,32]

    metric = [] # Search border x block_size

    pepn = []
    msen = []

    for res in results:
        pepn.append(res[7])
        msen.append(res[6])
        if res[0] == 'backward':
            if res[4] == "template2":
                # block_size.append(res[1])
                # search_border.append(res[2])
                metric.append(res[7]) # PEPN
                #metric.append(res[6]) # MSEN
    
    print(pepn)
    print(msen)

    pepn.sort()
    msen.sort()

    print("\n")
    print(pepn)
    print(msen)

    BLOCK, SEARCH = np.meshgrid(block_size, search_border)
    metric = np.reshape(metric, (5, 5))


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(BLOCK, SEARCH, metric, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('Block size')
    ax.set_ylabel('Search border')
    ax.set_zlabel('PEPN')
    ax.set_title('Backward (CCOEFF_NORM)')
    plt.show()
    
if __name__ == "__main__":
    #main()
    grid_search_plot3()