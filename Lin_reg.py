import random


def z_confidence_interval(data, st_dev, con_lvl):
    import statistics as st
    import scipy.stats as sp
    import math
    sample_mean = st.mean(data)
    n = len(data)
    crit_value = sp.norm.ppf(((1 - con_lvl) / 2) + con_lvl)
    lower_limit = sample_mean - (crit_value * (st_dev / math.sqrt(n)))
    higher_limit = sample_mean + (crit_value * (st_dev / math.sqrt(n)))
    print(f'Your {con_lvl} z confidence interval is ({lower_limit}, {higher_limit})')
    return lower_limit, higher_limit


sample_list = []
for i in range(30):
    sample_list.append(random.randint(0, 10))

z_confidence_interval(sample_list, 2.89, 0.95)
