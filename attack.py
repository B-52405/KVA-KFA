import random
import numpy as np
import math
import util
import time

random.seed(time.time())
np.random.seed(int(time.time() * 1000) % 2**32)

"""
对同一数据集每次baseline的结果都是随机的
对同一次baseline的结果执行不同的攻击以比较性能
对多次baseline后攻击的结果取平均值以消除随机性影响
"""
def baseline(
        dataset,
        keyword_leakage_rate,
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

def BVA(
        observed_queries,
        target_queries,
        keyword_real_sizes,
        keywords_leaked,
        recover_max):
    """
    inject
    """
    keyword_leaked_count = len(keywords_leaked)
    gamma = math.ceil(keyword_leaked_count / 2)
    keyword_injected_sizes = keyword_real_sizes.copy()
    index_injection_sizes = [index * gamma for index in range(keyword_leaked_count)]
    injection_size_to_index = {index_injection_sizes[i]:i for i in range(keyword_leaked_count)}

    for index in range(keyword_leaked_count):
        id = keywords_leaked[index]
        keyword_injected_sizes[id] += index_injection_sizes[index]

    """
    recover
    """
    recover_count = 0
    for query_target in target_queries:
        for query_observed in observed_queries:
            delta_size = keyword_injected_sizes[query_target] - keyword_real_sizes[query_observed]
            if delta_size not in injection_size_to_index:
                continue
            index = injection_size_to_index[delta_size]
            if keywords_leaked[index] == query_target:
                recover_count += 1   
            break 
    accuracy = recover_count / recover_max

    return accuracy

def KVA(
        observed_queries,
        target_queries,
        keyword_real_lengths,
        keyword_real_sizes,
        keywords_leaked,
        recover_max,
        k):
    """
    inject
    """
    keyword_leaked_count = len(keywords_leaked)
    gamma = math.ceil(keyword_leaked_count / 2)
    batch_size = math.ceil(keyword_leaked_count / k)
    keyword_injected_sizes = keyword_real_sizes.copy()
    keyword_injected_lengths = keyword_real_lengths.copy()
    injection_length_alpha = math.ceil(math.log2(batch_size))
    index_injection_lengths = [sum(1 if (index & (1 << i)) > 0 else 0 for i in range(injection_length_alpha)) for index in range(batch_size)]

    for index_in_batch in range(batch_size):
        for batch_index in range(k):
            index = index_in_batch + batch_size * batch_index
            if index >= keyword_leaked_count:
                break
            id = keywords_leaked[index]
            keyword_injected_sizes[id] += index_in_batch * gamma + 2 * gamma
            keyword_injected_lengths[id] += index_injection_lengths[index_in_batch] + (batch_index + 1)

    """
    recover
    """
    recover_count = 0
    for query_target in target_queries:
        for query_observed in observed_queries:
            delta_size = keyword_injected_sizes[query_target] - keyword_real_sizes[query_observed]
            if delta_size < 0 or delta_size % gamma != 0:
                continue
            delta_length = keyword_injected_lengths[query_target] - keyword_real_lengths[query_observed]
            if delta_length < 0:
                continue
            index_in_batch = (delta_size // gamma) - 2
            if index_in_batch >= batch_size:
                continue
            batch_index = delta_length - index_injection_lengths[index_in_batch] - 1
            if batch_index < 0 or batch_index >= k:
                continue
            index = index_in_batch + batch_size * batch_index
            if index >= keyword_leaked_count:
                continue
            if keywords_leaked[index] == query_target:
                recover_count += 1   
            break 
    accuracy = recover_count / recover_max
    
    return accuracy

def BVMA(
        observed_queries,
        target_queries,
        keyword_real_lengths,
        keyword_real_sizes,
        keywords_leaked,
        recover_max):
    """
    inject
    """
    keyword_leaked_count = len(keywords_leaked)
    gamma = math.ceil(keyword_leaked_count / 2)
    injection_length = math.ceil(math.log2(keyword_leaked_count))
    keyword_injected_sizes = keyword_real_sizes.copy()
    keyword_injected_lengths = keyword_real_lengths.copy()
    index_injection_lengths = [sum((index >> i) & 1 for i in range(injection_length)) for index in range(keyword_leaked_count)]
    index_injection_sizes = [gamma * index_injection_lengths[index] + index for index in range(keyword_leaked_count)]
    injection_size_to_index = {index_injection_sizes[i]:i for i in range(keyword_leaked_count)}

    for index in range(keyword_leaked_count):
        id = keywords_leaked[index]
        keyword_injected_sizes[id] += index_injection_sizes[index]
        keyword_injected_lengths[id] += index_injection_lengths[index]

    """
    recover
    """
    recover_count = 0
    for query_target in target_queries:
        for query_observed in observed_queries:
            delta_size = keyword_injected_sizes[query_target] - keyword_real_sizes[query_observed]
            if delta_size not in injection_size_to_index:
                continue
            index = injection_size_to_index[delta_size]
            delta_length = keyword_injected_lengths[query_target] - keyword_real_lengths[query_observed]
            if delta_length != index_injection_lengths[index]:
                continue
            if keywords_leaked[index] == query_target:
                recover_count += 1   
            break 
    accuracy = recover_count / recover_max

    return accuracy

def Decoding(
        observed_queries,
        target_queries,
        keyword_real_sizes,
        keywords_leaked,
        recover_max,
        offset):
    """
    inject
    """
    keyword_leaked_count = len(keywords_leaked)
    keyword_injected_sizes = keyword_real_sizes.copy()
    index_injection_sizes = [index * offset for index in range(keyword_leaked_count)]
    injection_size_to_index = {index_injection_sizes[i]:i for i in range(keyword_leaked_count)}

    for index in range(keyword_leaked_count):
        id = keywords_leaked[index]
        keyword_injected_sizes[id] += index_injection_sizes[index]

    """
    recover
    """
    recover_count = 0
    for query_target in target_queries:
        for query_observed in observed_queries:
            delta_size = keyword_injected_sizes[query_target] - keyword_real_sizes[query_observed]
            if delta_size not in injection_size_to_index:
                continue
            index = injection_size_to_index[delta_size]
            if keywords_leaked[index] == query_target:
                recover_count += 1   
            break 
    accuracy = recover_count / recover_max

    return accuracy
