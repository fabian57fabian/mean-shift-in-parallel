import matplotlib.pyplot as plt
import sys
import os
import csv
import pandas as pd

main_folder = "datas"


def print_all_dirs():
    for folder_pts in sorted(os.listdir(main_folder)):
        N = int(folder_pts)
        path = os.path.join(main_folder, folder_pts, 'points.csv')
        points = pd.read_csv(path)
        points.columns = ['X', 'Y']
        points.plot.scatter(x='X', y='Y', title=f'Dataset with {N} points')
        plt.show()


if __name__ == '__main__':
    print_all_dirs()
