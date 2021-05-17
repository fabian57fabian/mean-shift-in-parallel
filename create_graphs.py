import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

out_folder = 'images'
src_folder = 'results'

def create_plots():
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for file in os.listdir(src_folder):
        if file.endswith('.csv'):
            print()
            print(file)
            file_path = os.path.join(src_folder, file)
            df = pd.read_csv(file_path)
            print(df.head())
            df.transpose()[1:].plot(title=file[:-4])
            plt.show()


if __name__ == '__main__':
    create_plots()