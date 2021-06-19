import os
import csv
import sys


def insert_timings_cuda_to_results_folder(fn: str, min_t: int = 8):
    if not os.path.exists(fn):
        print("Given csv path does not exist: {}".format(fn))
        exit(1)
    # Create basic data structures
    tiles = []
    datas_naive, datas_sm = {}, {}
    datasets = os.listdir("datas")
    for d in datasets:
        datas_naive[int(d)] = []
        datas_sm[int(d)] = []
    #read results file
    with open(fn) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  #skip header
        for row in reader:
            if len(row) == 0: continue
            pts, tile_w, naive, sm = int(row[0]), int(row[1]), float(row[2]), float(row[3])
            if tile_w not in tiles: tiles.append(tile_w)
            datas_naive[pts].append(naive)
            datas_sm[pts].append(sm)

    with open("results/timings_cuda_naive.csv", 'w') as file_naive_out:
        spamwriter = csv.writer(file_naive_out, delimiter=',')
        header = ["dataset"]+["t{}".format(t) for t in tiles]
        spamwriter.writerow(header)
        for k, v in datas_naive.items():
            spamwriter.writerow([k]+v)

    with open("results/timings_cuda_shared.csv", 'w') as file_sm_out:
        spamwriter = csv.writer(file_sm_out, delimiter=',')
        spamwriter.writerow(["dataset"]+["t{}".format(t) for t in tiles])
        for k, v in datas_sm.items():
            spamwriter.writerow([k]+v)

    # write out file naive
    # write out file sm


if __name__ == '__main__':
    fn = "CUDA/MS_CUDA/results.csv"
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    insert_timings_cuda_to_results_folder(fn)
