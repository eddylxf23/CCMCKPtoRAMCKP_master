import math
import numpy as np
from scipy.stats import norm

def solution_evaluation(solution, item2obj, T, confidence_level, _lambda = 0, risk_averse = None, MonteCarlo = None):
    item_list = []
    for item_id in solution:
        item_list.append(item2obj[item_id])

    if risk_averse is None:
        # deterministic formulation of CC-MCKP
        s = sum([f.mean for f in item_list]) + norm.ppf(confidence_level) * math.sqrt(sum([f.variance for f in item_list]))
        if s <= T:
            result_sign = 1
        else:
            result_sign = (T-s)/T
    else:
        # RA-MCKP
        s = sum([f.mean for f in item_list]) + _lambda * sum([f.variance for f in item_list])
        if s <= T:
            result_sign = 1
        else:
            result_sign = (T-s)/T
    
    if MonteCarlo is not None:
        # simulation for original formulation of CC-MCKP
        estimated_cl = MC_solution_evaluation(item_list, T, 1000, int(MonteCarlo/1000))
        return result_sign, estimated_cl
    else:
        return result_sign


def Norm(sample_size,mu_,variance_):
    return np.random.normal(mu_, np.sqrt(variance_), sample_size)

def MC_solution_evaluation(item_list, Tmax, group_num, small_sample_size):
    total_ = 0

    for _ in range(group_num):
        sum_samples = np.sum(Norm(small_sample_size, item.mean, item.variance) for item in item_list)
        total_ += len(np.where(sum_samples <= Tmax)[0])

    # Calculate the ratio of sums which are smaller than Tmax across all samples
    return total_ / (group_num*small_sample_size)

def solution_initialization(factor_resort, param, order):
    factor_id_list = [i for i in range(int(param[0]))]   
    Init_solution = np.zeros(len(factor_id_list), dtype=int)

    for factor_id in factor_id_list:
        # 取出按照排好序的每个node中的第一个factor，即对应属性最大的factor
        Init_solution[factor_id] = factor_resort[factor_id][order]
    
    return Init_solution

def solution_cost(solution, item2obj):
    total_cost = 0

    for item_id in solution:
        item = item2obj[item_id]
        total_cost += item.cost
    
    return total_cost