import random
import os
import csv

main_folder = "datas"


def gauss_2d(mu, sigmas):
    x = random.gauss(mu[0], sigmas[0])
    y = random.gauss(mu[1], sigmas[1])
    return [x, y]


def random_item(_list):
    return random.choice(_list)


def generate_dataset(N, centroids, sigmas):
    folder = os.path.join(main_folder, str(N))
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, 'centroids.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        [writer.writerow(c) for c in centroids]
    with open(os.path.join(folder, 'points.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for _ in range(N):
            writer.writerow(gauss_2d(random_item(centroids), random_item(sigmas)))


if __name__ == "__main__":
    centroids = [[1, 9], [9, 1], [10, 9]]
    sigmas = [[1, 1.5] for _ in centroids]
    for N in [1000, 5000, 10000, 50000, 100000]:
        generate_dataset(N, centroids, sigmas)
