import util
from multiprocessing import Pool
import time
import attack


def single_round(_):
    accuracy_bvma = [0] * keyword_leakage_step_count
    accuracy_f_bvma = [0] * keyword_leakage_step_count
    accuracy_bva = [0] * keyword_leakage_step_count
    accuracy_f_bva = [0] * keyword_leakage_step_count
    
    keyword_leakage_rates = util.get_keyword_leakage_rates(keyword_leakage_step_count)
    for i, keyword_leakage_rate in enumerate(keyword_leakage_rates):
        (observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            keywords_leaked_sorted,
            recover_max) = attack.baseline(dataset, keyword_leakage_rate, document_leakage_rate)
        
        accuracy_bvma[i] = attack.BVMA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked,
            recover_max)

        accuracy_f_bvma[i] = attack.BVMA(
            observed_queries,
            target_queries,
            keyword_real_lengths,
            keyword_real_sizes,
            keywords_leaked_sorted,
            recover_max)
        
        accuracy_bva[i] = attack.BVA(
            observed_queries,
            target_queries,
            keyword_real_sizes,
            keywords_leaked,
            recover_max)
        
        accuracy_f_bva[i] = attack.BVA(
            observed_queries,
            target_queries,
            keyword_real_sizes,
            keywords_leaked_sorted,
            recover_max)

    return (
        accuracy_bvma, 
        accuracy_f_bvma, 
        accuracy_bva,
        accuracy_f_bva)


dataset_name = util.DATASET.lucene.name
dataset = util.load_dataset(dataset_name)
keyword_leakage_step_count = 10
keyword_leakage_rates = util.get_keyword_leakage_rates(keyword_leakage_step_count)
document_leakage_rate = 0.05
experiment_name = "f_method"
experiment_time = 30


if __name__ == "__main__":
    accuracy_bvma = [0] * keyword_leakage_step_count
    accuracy_f_bvma = [0] * keyword_leakage_step_count
    accuracy_bva = [0] * keyword_leakage_step_count
    accuracy_f_bva = [0] * keyword_leakage_step_count

    start_time = time.time()
    
    with Pool(processes = min(experiment_time, 12)) as pool:
        for result in pool.map(single_round, range(experiment_time)):
            (accuracy_bvma_temp, 
                accuracy_f_bvma_temp, 
                accuracy_bva_temp,
                accuracy_f_bva_temp) = result
            for i in range(keyword_leakage_step_count):
                accuracy_bvma[i] += accuracy_bvma_temp[i]
                accuracy_f_bvma[i] += accuracy_f_bvma_temp[i]
                accuracy_bva[i] += accuracy_bva_temp[i]
                accuracy_f_bva[i] += accuracy_f_bva_temp[i]

    for i in range(keyword_leakage_step_count):
        accuracy_bvma[i] /= experiment_time
        accuracy_f_bvma[i] /= experiment_time
        accuracy_bva[i] /= experiment_time
        accuracy_f_bva[i] /= experiment_time

    util.save_result(experiment_name, accuracy_bva, f"bva_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_bvma, f"bvma_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_f_bva, f"f_bva_{dataset_name}_{experiment_time}")
    util.save_result(experiment_name, accuracy_f_bvma, f"f_bvma_{dataset_name}_{experiment_time}")

    end_time = time.time()
    print(f"time: {end_time - start_time}")
