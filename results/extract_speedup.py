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


def compute_cuda_speedup(seq, cuda, dataset):
    speedups = []
    for p_value in cuda:
        speedups.append(float(seq/p_value))
    # end
    result = {'dataset': dataset, 't8': speedups[0], 't16': speedups[1], 't32': speedups[2],
              't64': speedups[3], 't128': speedups[4], 't256': speedups[5], 't512': speedups[6], 't1024': speedups[7]}
    return result
# end


def extract_cuda_speedup(dataset, sequential_timings, cuda_file, outputfile):
    with open(outputfile, 'w', newline='') as file:
        fieldnames = ['dataset', 't8', 't16', 't32',
                      't64', 't128', 't256', 't512', 't1024']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for d in range(len(dataset)):
            cuda_values = extract_cuda_values(cuda_file, dataset[d])
            writer.writerow(compute_cuda_speedup(
                sequential_timings[d], cuda_values, dataset[d]))
        # end
    # end
# end

def evaluate_speedup_dyn(dataset, sequential_timings, omp_timings, outputfile):
    with open(outputfile, 'w', newline='') as file:
        fieldnames = ['dataset', 'speedup']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for d in range(len(dataset)):
            speedup = {'dataset':dataset[d], 'speedup':sequential_timings[d]/omp_timings[d]}
            writer.writerow(speedup)
        # end
    # end
# end

def extract_omp_values(omp_file, dataset):
    timings = []
    with open(omp_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            if dataset == row['dataset']:
                timings.append(float(row['t1']))
                timings.append(float(row['t2']))
                timings.append(float(row['t3']))
                timings.append(float(row['t4']))
                timings.append(float(row['t5']))
                timings.append(float(row['t6']))
                timings.append(float(row['t7']))
                timings.append(float(row['t8']))
            # end
        # end
    # end
    return timings
# end

def compute_speedup_omp(seq, omp, dataset):
    speedups = []
    for p_value in omp:
        speedups.append(float(seq/p_value))
    # end
    result = {'dataset': dataset, 't1': speedups[0], 't2': speedups[1], 't3': speedups[2],
              't4': speedups[3], 't5': speedups[4], 't6': speedups[5], 't7': speedups[6], 't8': speedups[7]}
    return result
# end

def extract_omp_speedup(dataset, sequential_timings, omp_file, outputfile):
    with open(outputfile, 'w', newline='') as file:
        fieldnames = ['dataset', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for d in range(len(dataset)):
            omp_timing = extract_omp_values(omp_file, dataset[d])
            speedup = compute_speedup_omp(sequential_timings[d], omp_timing, dataset[d])
            writer.writerow(speedup)
        # end
    # end
# end


if __name__ == '__main__':
    dataset = ['500', '1000', '2000', '5000', '10000']
    sequential_timings = extract_seq('timings_seq.csv')

    # Speedup for omp_static version
    outputfile = "speedups_omp_static.csv"
    omp_file = 'timings_omp_static.csv'
    extract_omp_speedup(dataset, sequential_timings, omp_file, outputfile)

    # Speedup for omp_dynamic version
    outputfile = "speedups_omp_dynamic.csv"
    omp_timings = extract_seq('timings_omp_dynamic.csv')
    evaluate_speedup_dyn(dataset, sequential_timings, omp_timings, outputfile)

    # Speedup for naive version
    outputfile = "speedups_cuda_naive.csv"
    cuda_file = 'timings_cuda_naive.csv'
    extract_cuda_speedup(dataset, sequential_timings, cuda_file, outputfile)

    # Speedup for shared version
    outputfile = "speedups_cuda_shared.csv"
    cuda_file = 'timings_cuda_shared.csv'
    extract_cuda_speedup(dataset, sequential_timings, cuda_file, outputfile)

# end
