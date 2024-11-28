'''
Benchmark_generator.py
生成每个因子的高斯分布，在进行随机扰动后生成。

'''

import os
import math
import random
import numpy as np
from utils import my_file
from collections import OrderedDict

class Benchmark:
    
    def __init__(self,factor_num, item_num):
        self.item_num = item_num # The number of items in each class
        self.factor_num = factor_num    # The number of classes
        self.item_num_sum = self.factor_num * self.item_num  # Total number of items
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.save_folder = os.path.join(current_directory, f'benchmark/instance_{factor_num}_{item_num}_')
        my_file.create_folder(self.save_folder)


    def _save_attributes(self, factor_id, distribution,  mu_, variance_, select = None):
        
        attr_dict = OrderedDict()
        attr_dict['dist_type'] = distribution
        attr_dict['para_mean'] = mu_
        attr_dict['param_variance'] = variance_

        if select is not None:
            attr_dict['select'] = select
        filename_ = f'{factor_id}_dist.txt'
        with open(my_file.real_path_of(self.save_folder, filename_), 'w') as f:
            for k, v in attr_dict.items():
                f.write(f'{k} {v}\n')
        print(f"Factor ID {factor_id}, Distribution {distribution}, Mean {mu_}, Variance {variance_ }, Success!")
        
    #---------------- Gaussian ----------------
    def generate_Gaussian(self,item_id,mean_,var_):
        mu_ = mean_*(1 + 1.2*np.random.rand()-0.6)
        variance_ = var_*(1 + 1.8*np.random.rand()-0.9)

        self._save_attributes(item_id, 'Gaussian', mu_, variance_)

    #---------------- Randomly generate benchmark ----------------
    def generate_benchmark(self):
        for item_id in range(self.item_num_sum):          
            self.generate_Gaussian(item_id, 2000, 12000)

    #---------------- Randomly generate cost ----------------
    def generate_factor_link(self):
        factor_id_list = [i for i in range(self.factor_num)]
        item_id_list = [i for i in range(self.item_num_sum)]

        factor_to_item = {}
        for factor_id in factor_id_list:
            res_list = []
            for item_id in item_id_list:
                if item_id % self.factor_num  == factor_id:
                    res_list.append(item_id)
            factor_to_item[factor_id] = res_list
        for k in factor_to_item:
            print(f'{k}: {factor_to_item[k]}')

        my_file.save_pkl_in_repo(factor_to_item, self.save_folder, 'factor_to_item.pkl')

        weight_items_list=[]
        for item_id in item_id_list:
            filename_ = f'{item_id}_dist.txt'
            with open(my_file.real_path_of(self.save_folder, filename_), 'r') as f:
                data = f.readlines()

            items_mean = float(data[1].strip().split()[1])
            items_variance = float(data[2].strip().split()[1])
            weight_items_list.append(items_mean + math.sqrt(items_variance))

        # 方式1： random
        item_cost_list = np.random.random(size=self.item_num_sum) * 90 + 50
        np.savetxt(my_file.real_path_of(self.save_folder, 'cost.txt'), item_cost_list)
        # 方式2：随着节点性能的提高，性价比在逐步降低, 并且加上一个随机偏置量
        # item_cost_list = [(100000/weight_items_list[i])*(1 + 0.4*np.random.rand()-0.2) for i in range(self.item_num_sum)]
        

if __name__ == '__main__':

    random.seed(827)

    Benchmark_1 = Benchmark(factor_num=100,item_num=50)
    Benchmark_1.generate_benchmark()
    Benchmark_1.generate_factor_link()


