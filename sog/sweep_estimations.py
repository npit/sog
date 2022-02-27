from collections import defaultdict
from statistics import mean
from random import shuffle, randint, choice
import pandas as pd


def measure_hits(world_size, mx):
    # print('mx', mx)
    n1, n2 = randint(1,mx), randint(1,mx)

    # print(n1, n2)

    # expected when random
    w1, w2 = [0 for _ in range(world_size)], [0 for _ in range(world_size)]

    w1[:n1] = [1] * n1
    w2[:n2] = [1] * n2
    shuffle(w1)
    shuffle(w2)
    # print(w1)
    # print(w2)

    hits = 0
    iters = 0
    while sum(w1) + sum(w2) > 0:
        for w in (w1, w2):
            iters += 1
            m = choice(range(len(w)))
            if w[m]:
                hits += 1
            w[m] = 0

        joint = w1 + w2
        shuffle(joint)
        w1, w2 = joint[:len(w1)], joint[len(w1):]
    return iters, hits

data = defaultdict(dict)
for world_size in range(5, 75, 5):
    data[world_size]= defaultdict(list)
    # stats
    for _ in range(25):
        mx = world_size*2//3
        iters, hits = measure_hits(world_size, mx)
        data[world_size]['iter'].append(iters)
        data[world_size]['hits'].append(hits)
        data[world_size]['misses'].append(iters-hits)
    for k in list(data[world_size].keys()):
        data[world_size][k] = mean(data[world_size][k]) / 2
        if k != 'iter':
            data[world_size][k + '_ratio'] = data[world_size][k] / data[world_size]['iter']


df = pd.DataFrame.from_dict(data, orient='index')
df.index.name = 'world size'
print(df)
