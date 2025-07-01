import util
import attack
from multiprocessing import Pool
import time


def single_round(_):
    accuracy_bva_rlp = [0] * keyword_leakage_step_count
    accuracy_kva_2 = [0] * keyword_leakage_step_count
    accuracy_kva_5 = [0] * keyword_leakage_step_count
    accuracy_kva_10 = [0] * keyword_leakage_step_count

    for i, keyword_leakage_rate in enumerate(keyword_leakage_rates):
        (observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            keywords_leaked,
            recover_max) = attack.baseline(dataset, keyword_leakage_rate)
        
        accuracy_bva_rlp[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max,
            1)
        
        accuracy_kva_2[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max,
            2)
        
        accuracy_kva_5[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max,
            5)
        
        accuracy_kva_10[i] = attack.KVA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max,
            10)
        
    return (
        accuracy_bva_rlp,
        accuracy_kva_2,
        accuracy_kva_5,
        accuracy_kva_10)


dataset_name = util.DATASET.enron.name
dataset = util.load_dataset(dataset_name)
keyword_leakage_step_count = 50
keyword_leakage_rates = util.get_keyword_leakage_rates(keyword_leakage_step_count)
experiment_name = "k_method"
experiment_time = 30


if __name__ == "__main__":
    accuracy_bva_rlp = [0] * keyword_leakage_step_count
    accuracy_kva_2 = [0] * keyword_leakage_step_count
    accuracy_kva_5 = [0] * keyword_leakage_step_count
    accuracy_kva_10 = [0] * keyword_leakage_step_count

    start_time = time.time()
    
    with Pool(processes = min(experiment_time, 12)) as pool:
        for result in pool.map(single_round, range(experiment_time)):
            (accuracy_bva_rlp_temp,
                accuracy_kva_2_temp,
                accuracy_kva_5_temp,
                accuracy_kva_10_temp) = result
            for i in range(keyword_leakage_step_count):
                accuracy_bva_rlp[i] += accuracy_bva_rlp_temp[i]
                accuracy_kva_2[i] += accuracy_kva_2_temp[i]
                accuracy_kva_5[i] += accuracy_kva_5_temp[i]
                accuracy_kva_10[i] += accuracy_kva_10_temp[i]

    for i in range(keyword_leakage_step_count):
        accuracy_bva_rlp[i] /= experiment_time
        accuracy_kva_2[i] /= experiment_time
        accuracy_kva_5[i] /= experiment_time
        accuracy_kva_10[i] /= experiment_time

    util.save_result(experiment_name, accuracy_bva_rlp, f"bva_rlp_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_kva_2, f"k_bva_2_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_kva_5, f"k_bva_5_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_kva_10, f"k_bva_10_{dataset_name}_{experiment_time}")

    end_time = time.time()
    print(f"time: {end_time - start_time}")
