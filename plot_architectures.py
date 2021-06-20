import os
import pandas as pd
import matplotlib.pyplot as plt

out_folder = "plots/MERGED"
show_plots = False

def get_col_trasposed(df1):
    p = df1.transpose()
    vals = p[p.columns[-1]]
    return vals

def create_combined(transpose:bool, debug=False):
    tw = "t64"
    pd_1n = pd.read_csv("results/GTX1050TI/speedups_cuda_naive.csv")
    pd_1s = pd.read_csv("results/GTX1050TI/speedups_cuda_shared.csv")
    pd_2n = pd.read_csv("results/GTX1660TI/speedups_cuda_naive.csv")
    pd_2s = pd.read_csv("results/GTX1660TI/speedups_cuda_shared.csv")
    pd_3n = pd.read_csv("results/RTX2060/speedups_cuda_naive.csv")
    pd_3s = pd.read_csv("results/RTX2060/speedups_cuda_shared.csv")

    df = pd.DataFrame()
    df["GTX1050TI naive"] = get_col_trasposed(pd_1n)
    df["GTX1050TI sm"] = get_col_trasposed(pd_1s)
    df["GTX1660TI naive"] = get_col_trasposed(pd_2n)
    df["GTX1660TI sm"] = get_col_trasposed(pd_2s)
    df["RTX2060 naive"] = get_col_trasposed(pd_3n)
    df["RTX2060 sm"] = get_col_trasposed(pd_3s)
    df = df.iloc[1:]#remove dataset row
    #df.set_index("dataset", inplace=True)
    if debug: print(df.head())
    if transpose: df = df.transpose()
    if debug: print(df.head())
    if debug: print(df.columns)

    ax = df.plot(title="Naive vs Optimized on dataset=10000", marker='.', markersize=10, figsize=(12, 7))
    x_label = "GPU type and version" if transpose else "Tile Width"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Speedup value")
    if show_plots: plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(out_folder, "speedups_cuda_combined{}.jpg".format("_by_dataset" if transpose else "_by_t" )))


if __name__ == '__main__':
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    create_combined(transpose=True)
    create_combined(transpose=False)