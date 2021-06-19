import os
import csv
import sys


def insert_timings_cuda_to_results_folder(fn: str, min_t: int = 8):
    if not os.path.exists(fn):
        print("Given csv path does not exist: {}".format(fn))
        exit(1)
    header = ""
    # THREADS,POINTS_NUMBER,NAIVE,SHARED
    with open(fn) as csvfile:
        reader = csv.reader(csvfile)
        first = True
        for row in reader:
            if first:
                header = row
            else:
                th, pts, naive, sm = row
                sm = sm.strp()
                # TODO: add accordingly
    # write out file naive
    # write out file sm


if __name__ == '__main__':
    fn = "CUDA/MS_CUDA/results.csv"
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    insert_timings_cuda_to_results_folder(fn)
