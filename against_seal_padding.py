import random
import numpy as np
import util
from multiprocessing import Pool
import attack
import bisect
import time


def baseline(
        dataset,
        x,
        keyword_leakage_rate = 1,
        document_leakage_rate = 0.1, 
        observed_period = 8,
        target_period = 10,
        number_queries_per_period = 1000):
    documents, keywords_dict = dataset
    keyword_count = len(keywords_dict)

    """
    baseline
    """
    observed_queries = []
    target_queries = []
    keyword_real_lengths = [0] * keyword_count
    keyword_real_sizes = [0] * keyword_count
    keywords_leaked = []
    keywords_leaked_sorted = []
    recover_max = 0

    keywords = np.random.permutation(list(keywords_dict.keys()))
    keywords_set = set(keywords)
    keyword_trends = np.vstack([keywords_dict[keyword]["trend"] for keyword in keywords])
    documents_leaked = random.sample(documents, int(len(documents) * document_leakage_rate))
    keyword_leaked_count = int(keyword_count * keyword_leakage_rate)
    keyword_leaked_sizes = [0] * keyword_count
    keywords_leaked = random.sample([i for i in range(keyword_count)], keyword_leaked_count)
    keyword_to_id = {keywords[i]: i for i in range(keyword_count)}

    for i in range(observed_period):
        trends = keyword_trends[:, i]
        trends = trends / np.sum(trends)
        queries = list(np.random.choice(keyword_count, number_queries_per_period, p = trends))
        observed_queries.extend([int(query) for query in queries])
    for i in range(observed_period, observed_period + target_period):
        trends = keyword_trends[:, i]
        trends = trends / np.sum(trends)
        queries = list(np.random.choice(keyword_count, number_queries_per_period, p = trends))
        target_queries.extend([int(query) for query in queries])

    keywords_leaked_set = [False] * keyword_count
    for id in keywords_leaked:
        keywords_leaked_set[id] = True
    recover_max = sum(1 if keywords_leaked_set[query] else 0 for query in target_queries)

    for document in documents:
        document_keyword_set = keywords_set.intersection(document)
        for id in document_keyword_set:
            keyword_real_sizes[keyword_to_id[id]] += len(document)
            keyword_real_lengths[keyword_to_id[id]] += 1
    for document in documents_leaked:
        document_keyword_set = keywords_set.intersection(document)
        for id in document_keyword_set:
            keyword_leaked_sizes[keyword_to_id[id]] += len(document)

    lengths = [x]
    while lengths[-1] < len(documents):
        lengths.append(lengths[-1] * x)
    document_size_min = 10 ** 9
    document_size_max = 0
    for document in documents:
        document_size_min = min(document_size_min, len(document))
        document_size_max = max(document_size_max, len(document))
    for id in keywords_leaked:
        target_length = lengths[bisect.bisect_left(lengths, keyword_real_lengths[id])]
        keyword_real_lengths[id] = target_length
        for _ in range(target_length - keyword_real_lengths[id]):
            keyword_real_sizes[id] += random.randint(document_size_min, document_size_max)

    keywords_leaked_sorted = sorted(keywords_leaked, key=lambda id:keyword_leaked_sizes[id])

    return (
        observed_queries,
        target_queries,
        keyword_real_lengths,
        keyword_real_sizes,
        keywords_leaked,
        keywords_leaked_sorted,
        recover_max
    )


def single_round(_):
    accuracy_kva = [0] * len(xs)
    accuracy_kfa = [0] * len(xs)
    accuracy_bva = [0] * len(xs)
    accuracy_bvma = [0] * len(xs)
    
    for i, x in enumerate(xs):
        (observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            keywords_leaked_sorted,
            recover_max) = baseline(dataset, x)
        
        accuracy_kva[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_sizes,
            keyword_real_sizes,
            keywords_leaked,
            recover_max,
            10)

        accuracy_kfa[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked_sorted,
            recover_max,
            10)
        
        accuracy_bva[i] = attack.BVA(
            observed_queries,
            target_queries,
            keyword_real_sizes,
            keywords_leaked,
            recover_max)
        
        accuracy_bvma[i] = attack.BVMA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max) 

    return (
        accuracy_kva,
        accuracy_kfa, 
        accuracy_bva,
        accuracy_bvma)


dataset_name = util.DATASET.enron.name
dataset = util.load_dataset(dataset_name)
xs = [2, 4, 16]
experiment_name = "against_seal_padding"
experiment_time = 30


if __name__ == "__main__":
    accuracy_kva = [0] * len(xs)
    accuracy_kfa = [0] * len(xs)
    accuracy_bva = [0] * len(xs)
    accuracy_bvma = [0] * len(xs)

    start_time = time.time()
    
    with Pool(processes = min(experiment_time, 12)) as pool:
        for result in pool.map(single_round, range(experiment_time)):
            (accuracy_kva_temp, 
                accuracy_kfa_temp, 
                accuracy_bva_temp,
                accuracy_bvma_temp) = result
            for i in range(len(xs)):
                accuracy_kva[i] += accuracy_kva_temp[i]
                accuracy_kfa[i] += accuracy_kfa_temp[i]
                accuracy_bva[i] += accuracy_bva_temp[i]
                accuracy_bvma[i] += accuracy_bvma_temp[i]

    for i in range(len(xs)):
        accuracy_kva[i] /= experiment_time
        accuracy_kfa[i] /= experiment_time
        accuracy_bva[i] /= experiment_time
        accuracy_bvma[i] /= experiment_time

    util.save_result(experiment_name, accuracy_kva, f"kva_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_kfa, f"kfa_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_bva, f"bva_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_bvma, f"bvma_{dataset_name}_{experiment_time}")

    end_time = time.time()
    print(f"time: {end_time - start_time}")
