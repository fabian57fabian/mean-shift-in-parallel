import os
import pandas as pd
import matplotlib.pyplot as plt

gpu_folder_name = "GTX1050TI"
gpu_folder_name2 = "GTX1660TI"
out_folder = "plots/MERGED"
show_plots = False


def create_combined_omp(debug=False):
    pd_1 = pd.read_csv("results/{}/speedups_omp_dynamic.csv".format(gpu_folder_name))
    pd_2 = pd.read_csv("results/{}/speedups_omp_dynamic.csv".format(gpu_folder_name2))

    df = pd.DataFrame(pd_1["dataset"])
    df["GTX1050TI"] = pd_1["speedup"]
    df["GTX1660TI"] = pd_2["speedup"]

    df.set_index("dataset", inplace=True)
    if debug: print(df.head())
    #df = df.transpose()
    if debug: print(df.head())
    if debug: print(df.columns)

    ax = df.plot(title="Speedups omp dynamic in different CPUs", marker='.', markersize=10, figsize=(12, 7))
    x_label = "Dataset size"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Speedup value")
    if show_plots: plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(out_folder, "speedups_omp_combined{}.jpg".format("_by_dataset")))


if __name__ == '__main__':
    create_combined_omp()