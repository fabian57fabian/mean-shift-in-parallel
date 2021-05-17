import os
import csv
import pandas as pd

out_folder = 'images'
src_folder = 'results'

def create_plots():
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for file in os.listdir(src_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(src_folder, file)
            df = pd.read_csv(file_path)
            pass


if __name__ == '__main__':
    create_plots()