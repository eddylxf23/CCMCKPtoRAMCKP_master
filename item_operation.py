import math
import numpy as np
from utils import my_file
from utils.item import Item

def max_var(factors_to_items):
    # Calculate the maximum of variance from all item combinations
    factor_num = len(factors_to_items)
    factor_id_list = [i for i in range(factor_num)]
    max_var_list = []
    for factor_id in factor_id_list:
        cur_factors = factors_to_items[factor_id]
        max_var_list.append(max([f.variance for f in cur_factors]))

    return sum(max_var_list)

def Get_item(param, _lambda, data_folder):
    factor_num, item_num = int(param[0]), int(param[1])
    item_num_sum = factor_num * item_num
    factor_id_list = [i for i in range(factor_num)]
    item_id_list = [i for i in range(item_num_sum)]

    # 读取节点与因子关系
    factor_to_item_id = my_file.load_pkl_in_repo(data_folder, 'factor_to_item.pkl')
    cost_list = np.loadtxt(my_file.real_path_of(data_folder, 'cost.txt'))
    max_cost = np.max(cost_list)

    item2obj = [] # 映射：因子id->具体因子
    for item_id in item_id_list:
        cost_ = cost_list[item_id]
        value_ = max_cost - cost_list[item_id]
        filename_ = f'{item_id}_dist.txt'
        with open(my_file.real_path_of(data_folder, filename_), 'r') as f:
            data = f.readlines()
        _mean = float(data[1].strip().split()[1])
        _variance = float(data[2].strip().split()[1])

        cur_item = Item(item_id, cost_, value_, _mean, _variance)
        item2obj.append(cur_item)

    factor_to_items = {factor_id : [] for factor_id in factor_id_list} # list 每个环节对应因子
    for factor_id in factor_id_list:
        cur_item_id_list = factor_to_item_id[factor_id]

        for cur_item_id in cur_item_id_list:
            factor_to_items[factor_id].append(item2obj[cur_item_id])

    factor_resort_utility = {factor_id: [] for factor_id in factor_id_list}
    factor_resort_weight = {factor_id: [] for factor_id in factor_id_list}
    factor_resort_value = {factor_id: [] for factor_id in factor_id_list}

    for factor_id in factor_id_list:
        cur_items = factor_to_items[factor_id]
        cur_items_mean_list = [f.mean for f in cur_items]
        cur_items_std_list = [math.sqrt(f.variance) for f in cur_items]
        cur_items_value_list = [f.value for f in cur_items]
        cur_items_std_list[:] = [x * _lambda for x in cur_items_std_list]
        
        # 样本均值和样本标准差按权重相加
        weight_items_list = [cur_items_mean_list[i] + cur_items_std_list[i] for i in range(len(cur_items))]
        utility_items_list = [cur_items_value_list[i] / weight_items_list[i] for i in range(len(cur_items))]

        for i in range(len(weight_items_list)):
            cur_items[i].set_weight(weight_items_list[i])
            cur_items[i].set_utility(utility_items_list[i])

        cand_indices_weight = np.argsort(weight_items_list)[::-1]    # 根据统计量构造的 weight=mean+lambda*std 从大到小 将每个class的因子重新排序
        cand_indices_utility = np.argsort(utility_items_list)[::-1]    # 根据统计量构造的 utility=value/weight 从大到小 将每个class的因子重新排序
        cand_indices_value = np.argsort(cur_items_value_list)[::-1]   # 根据 value 构造的 从大到小 将每个class的因子重新排序

        for _idx in cand_indices_weight:
            factor_resort_weight[factor_id].append(cur_items[_idx].id)

        for _idx in cand_indices_utility:
            factor_resort_utility[factor_id].append(cur_items[_idx].id)

        for _idx in cand_indices_value:
            factor_resort_value[factor_id].append(cur_items[_idx].id)

    value_list = []
    for factor_id in factor_id_list:
        cur_items = factor_to_items[factor_id]
        cur_items_value_list = [f.value for f in cur_items]
        value_list.append(cur_items_value_list)
    
    return value_list, factor_to_items, item2obj, factor_resort_weight, factor_resort_utility, factor_resort_value

def Get_weight(factor_to_items, _lambda):

    factor_num = len(factor_to_items)
    factor_id_list = [i for i in range(factor_num)]
    weight_list = []

    for factor_id in factor_id_list:
        cur_items = factor_to_items[factor_id]
        cur_items_mean_list = [item.mean for item in cur_items]
        cur_items_var_list = [item.variance for item in cur_items]
        lambda_va = [x * _lambda for x in cur_items_var_list]
        
        # 样本均值和样本标准差按权重相加
        weight_items_list = [cur_items_mean_list[i] + lambda_va[i] for i in range(len(cur_items))]
        weight_list.append(weight_items_list)
    
    return weight_list

if __name__ == '__main__':

    folder_name = 'benchmark/instance_4_4_'
    # param = re.findall(r'\d+',folder_name)
    # Tmax = 14
    # CL = 0.9
    # MaxIter_LS = 30
    # _lambda = 1

