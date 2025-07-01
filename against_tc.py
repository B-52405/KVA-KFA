import numpy as np
import math
import util


keyword_counts = [i for i in range(0, 50001, 5000)]
file_sizes = [1000 * (2 ** i) for i in range(11)]
X, Y = np.meshgrid(keyword_counts, file_sizes)
experiment_name = "against_tc"
T = 200


def gtm(keyword_count, file_size, T):
    inject_length = math.ceil(keyword_count / T) * math.floor(file_size / T)
    if file_size % T != 0:
        inject_length += math.ceil(T / (file_size % T))
    return inject_length

def agtm(keyword_count, file_sizes, T):
    return math.ceil(keyword_count / T)


inject_lengths_gtm = np.array([[gtm(keyword_count, file_size, T) for keyword_count in keyword_counts] for file_size in file_sizes])
inject_lengths_agtm = np.array([[agtm(keyword_count, file_size, T) for keyword_count in keyword_counts] for file_size in file_sizes])

util.save_result(experiment_name, inject_lengths_agtm, "inject_lengths_agtm")
util.save_result(experiment_name, inject_lengths_gtm, "inject_lengths_gtm")
