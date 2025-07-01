import pickle
import json
import os
import itertools
import math
from enum import Enum
from datetime import datetime


class PATH(Enum):
    dataset = 1
    result = 2
    image = 3


class DATASET(Enum):
    enron = 1
    lucene = 2
    wikipedia = 3


def format_time():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def smallest_divisible(a, b):
    return a + (b - a % b) % b


def load_dataset(dataset_name):
    
    with open(os.path.join(PATH.dataset.name, f"{dataset_name}_doc.pkl"), "rb") as f:
        documents = pickle.load(f)
    with open(os.path.join(PATH.dataset.name, f"{dataset_name}_kws_dict.pkl"), "rb") as f:
        keywords_dict = pickle.load(f)
    return documents, keywords_dict


def save_json(data, path):
    with open(os.path.join('JSON', path), 'w') as f:
        json.dump(data, f)


combs = [[math.comb(m, n) for n in range(m // 2 + 1)] for m in range(21)]
def generate_codes(count):
    def generate_m_n(count):
        for m in range(1, 21):
            for n in range(1, m // 2 + 1):
                if combs[m][n] >= count:
                    return (m, n)
    m, n = generate_m_n(count)
    codes = []
    for bits in itertools.combinations(range(m), n):
        code = sum(1 << index for index in bits)
        codes.append(code)
    codes = sorted(codes)[:count]  
    return m, n, codes


def get_keyword_leakage_rates(count):
    step = 1 / count
    keyword_leakage_rates = [step * i for i in range(1, count + 1)]
    return keyword_leakage_rates


def save_result(experiment_name, result, result_name):
    with open(os.path.join(PATH.result.name, experiment_name, f"{result_name}.pkl"), "wb") as f:
        pickle.dump(result, f)

def load_result(experiment_name, result_name):
    with open(os.path.join(PATH.result.name, experiment_name, f"{result_name}.pkl"), "rb") as f:
        data = pickle.load(f)
    return data
