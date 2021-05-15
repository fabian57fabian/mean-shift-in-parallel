import numpy as np
import csv


def extract_seq(results_file):
    timings = []
    with open(results_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            timings.append(float(row['time']))
        # end
    # end
    return timings
# end


def extract_cuda_values(results_file, dataset):
    timings = []
    with open(results_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            if dataset == row['dataset']:
                timings.append(float(row['t8']))
                timings.append(float(row['t16']))
                timings.append(float(row['t32']))
                timings.append(float(row['t64']))
                timings.append(float(row['t128']))
                timings.append(float(row['t256']))
                timings.append(float(row['t512']))
                timings.append(float(row['t1024']))
            # end
        # end
    # end
    return timings
# end


def compute_speedup(seq, cuda, dataset):
    speedups = []
    for p_value in cuda:
        speedups.append(float(seq/p_value))
    # end
    result = {'dataset': dataset, 't8': speedups[0], 't16': speedups[1], 't32': speedups[2],
              't64': speedups[3], 't128': speedups[4], 't256': speedups[5], 't512': speedups[6], 't1024': speedups[7]}
    return result
# end


def extract_speedup(dataset, sequential_timings, cuda_file, outputfile):
    with open(outputfile, 'w', newline='') as file:
        fieldnames = ['dataset', 't8', 't16', 't32',
                      't64', 't128', 't256', 't512', 't1024']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for d in range(len(dataset)):
            cuda_values = extract_cuda_values(cuda_file, dataset[d])
            writer.writerow(compute_speedup(
                sequential_timings[d], cuda_values, dataset[d]))
        # end
    # end
# end


if __name__ == '__main__':
    dataset = ['500', '1000', '2000', '5000', '10000']
    sequential_timings = extract_seq('timings_seq.csv')

    # Speedup for naive version
    outputfile = "speedups_cuda_naive.csv"
    cuda_file = 'timings_cuda_naive.csv'
    extract_speedup(dataset, sequential_timings, cuda_file, outputfile)

    # Speedup for shared version
    outputfile = "speedups_cuda_shared.csv"
    cuda_file = 'timings_cuda_shared.csv'
    extract_speedup(dataset, sequential_timings, cuda_file, outputfile)
    
# end
