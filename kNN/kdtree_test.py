from kdtree_search import *
from time import clock
from random import random

# k-dimensional random vector
def random_vector(k):
    return [random() * 100 for _ in range(k)]

if __name__ == '__main__':
    N = 40000
    vecs = [random_vector(3) for _ in range(N)]
    label = [random() for _ in range(N)]
    kd = KdTree(vecs, label)
    t0 = clock()
    ret = find_nearest(kd, [10, 50, 80])
    t1 = clock()
    print('Time: ', t1 - t0)
    print(ret)