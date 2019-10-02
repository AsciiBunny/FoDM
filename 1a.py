import math
from functools import reduce


def setup_clusters(initial_centroids):
    clusters = dict()
    for i, centroid in enumerate(initial_centroids):
        clusters[f'Cl{i}'] = {
            'centroid': centroid,
            'cluster': set(),
            'previous_cluster': []
        }
    return clusters

def reset_clusters(clusters):
    for c in clusters:
        clusters[c]['previous_cluster'] = clusters[c]['cluster']
        clusters[c]['cluster'] = set()


def euclidean(a, b):
    return math.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))


def assign_clusters(clusters, data):
    for a in data:
        min_dst = math.inf
        min_c = ''
        for c in clusters:
            dst = euclidean(data[a], clusters[c]['centroid'])
            if dst < min_dst:
                min_dst = dst
                min_c = c
            elif dst == min_dst:
                min_c = c if euclidean((0, 0), clusters[c]['centroid']) < euclidean((0, 0), clusters[min_c][
                    'centroid']) else min_c
        clusters[min_c]['cluster'].add(a)


def compute_centroids(clusters, data):
    for c in clusters:
        cluster = clusters[c]['cluster']
        count = len(cluster)
        sum = reduce((lambda x, y: (x[0] + data[y][0], x[1] + data[y][1])), cluster, (0, 0))
        clusters[c]['centroid'] = (sum[0] / count, sum[1] / count)


def check_convergence(clusters):
    for c in clusters:
        if clusters[c]['previous_cluster'] == clusters[c]['cluster']:
            continue
        else:
            return False
    return True


def print_clusters(clusters):
    for c in clusters:
        print(c, clusters[c]['centroid'], clusters[c]['cluster'])


def calculate_within_cluster_distance(clusters, data):
    sum = 0
    for c in clusters:
        cluster = clusters[c]['cluster']
        for a in cluster:
            for b in cluster:
                sum = sum + euclidean(data[a], data[b])
    return sum / 2


def calculate_total_distance(data):
    sum = 0
    for a in data:
        for b in data:
            sum = sum + euclidean(data[a], data[b])
    return sum / 2


def k_means(data, initial_centroids):
    clusters = setup_clusters(initial_centroids)
    while not check_convergence(clusters):
        reset_clusters(clusters)
        assign_clusters(clusters, data)
        compute_centroids(clusters, data)
    within_distance = calculate_within_cluster_distance(clusters, data)
    total_distance = calculate_total_distance(data)
    print('Within-cluster Distance:', within_distance)
    print('Between-cluster Distance', total_distance - within_distance)
    print_clusters(clusters)


omega = {
    'A': (0, 9),
    'B': (1, 8),
    'C': (1, 5),
    'D': (3, 1),
    'E': (5, 2),
    'F': (5, 0),
    'G': (8, 5),
    'H': (11, 5),
}

print("Initial Centroids: A B C")
k_means(omega, [omega['A'], omega['B'], omega['C']])
print()
print("Initial Centroids: E F G")
k_means(omega, [omega['E'], omega['F'], omega['G']])