import util
import attack
from multiprocessing import Pool
import time


def single_round(_):
    accuracy_kva = [0] * keyword_leakage_step_count
    accuracy_kfa = [0] * keyword_leakage_step_count
    accuracy_bva = [0] * keyword_leakage_step_count
    accuracy_bvma = [0] * keyword_leakage_step_count
    accuracy_decoding = [0] * keyword_leakage_step_count

    for i, keyword_leakage_rate in enumerate(keyword_leakage_rates):
        (observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            keywords_leaked_sorted,
            recover_max) = attack.baseline(dataset, keyword_leakage_rate)

        accuracy_kva[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max,
            32)
        
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
        
        accuracy_decoding[i] = attack.Decoding(
            observed_queries,
            target_queries,
            keyword_real_sizes,
            keywords_leaked,
            recover_max)
        
    return (
        accuracy_kva,
        accuracy_kfa,
        accuracy_bvma,
        accuracy_bva,
        accuracy_decoding)


dataset_name = util.DATASET.lucene.name
dataset = util.load_dataset(dataset_name)
keyword_leakage_step_count = 50
keyword_leakage_rates = util.get_keyword_leakage_rates(keyword_leakage_step_count)
experiment_name = "against_other_attacks"
experiment_time = 30


if __name__ == "__main__":
    accuracy_kva = [0] * keyword_leakage_step_count
    accuracy_kfa = [0] * keyword_leakage_step_count
    accuracy_bva = [0] * keyword_leakage_step_count
    accuracy_bvma = [0] * keyword_leakage_step_count
    accuracy_decoding = [0] * keyword_leakage_step_count

    start_time = time.time()
    
    with Pool(processes = min(experiment_time, 12)) as pool:
        for result in pool.map(single_round, range(experiment_time)):
            (accuracy_kva_temp,
                accuracy_kfa_temp,
                accuracy_bva_temp,
                accuracy_bvma_temp,
                accuracy_decoding_temp) = result
            for i in range(keyword_leakage_step_count):
                accuracy_kva[i] += accuracy_kva_temp[i]
                accuracy_kfa[i] += accuracy_kfa_temp[i]
                accuracy_bva[i] += accuracy_bva_temp[i]
                accuracy_bvma[i] += accuracy_bvma_temp[i]
                accuracy_decoding[i] += accuracy_decoding_temp[i]

    for i in range(keyword_leakage_step_count):
        accuracy_kva[i] /= experiment_time
        accuracy_kfa[i] /= experiment_time
        accuracy_bvma[i] /= experiment_time
        accuracy_bva[i] /= experiment_time
        accuracy_decoding[i] /= experiment_time

    util.save_result(experiment_name, accuracy_kva, f"kva_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_kfa, f"kfa_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_bva, f"bva_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_bvma, f"bvma_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_decoding, f"decoding_{dataset_name}_{experiment_time}")

    end_time = time.time()
    print(f"time: {end_time - start_time}")
