'''
main function for

Xuanfeng Li, Shengcai Liu, and Ke Tang. Novel Genetic Algorithm for Solving Chance-Constrained Multiple-Choice Knapsack Problems. 
Journal of Computer Applications, 2024, 44(5): 1378-1385

'''


import re
import csv
import random
import time
from typing import List
from utils import my_file
from multiprocessing import Process as _Process
from main import CCMCKP_main

def main_exp(ps, pc, es, evastop, times, Tmax, CL, param, algorithm, folder_name, file_):
    Elitesize = int(ps*es) 
    mi = 1
    optimal_solution, optimal_obj, confilevel, runtime, eval_times, solve_ra_times =  CCMCKP_main(ps, pc, Elitesize, evastop, mi, Tmax, CL, param, algorithm, folder_name)
    print(f"\n{algorithm} 最优解为:", optimal_solution, "最优成本为:", optimal_obj, "运行时间为:", runtime, "评估结果：", confilevel, "Evaluation次数:", eval_times, "RA求解次数:", solve_ra_times)
    record_to_csv(file_, algorithm, times, optimal_obj, confilevel, runtime, eval_times, solve_ra_times)  

def create_csv(file_, algorithm):  
    path1 = my_file.real_path_of(f'CCMCKPtoRAMCKP/results/{algorithm}/{file_}')
    with open(path1,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Times', 'Cost', 'Confidence Level', 'Runtime', 'Evaluation Times', 'RA solved times']
        csv_write.writerow(csv_head)

def record_to_csv(file_, algorithm, t, obj, cl, rt, et, srt):
    path = my_file.real_path_of(f'CCMCKPtoRAMCKP/results/{algorithm}/{file_}')
    with open(path,'a',newline='') as f:
        csv_write = csv.writer(f)
        data_row = [t, obj, cl, rt, et, srt]
        csv_write.writerow(data_row)

if __name__=="__main__":
    se = 827323
    Iterative_time = 30
    CL = 0.99
    random.seed(se)
    # algorithm = ["LS","GA","LS-GA","DP"]
    algorithm = ["GA"]
    maxiter_ = [200,300,500]
    populationsize = [200,300,500]
    pc = 0.6
    es = 0.6

    process_list: List[_Process] = []

    for algo in algorithm:
        instance_folder = [\
                'CCMCKPtoRAMCKP/benchmark/instance_3_3_',
                'CCMCKPtoRAMCKP/benchmark/instance_3_5_',
                'CCMCKPtoRAMCKP/benchmark/instance_3_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_3_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_5_3_',
                'CCMCKPtoRAMCKP/benchmark/instance_5_5_',
                'CCMCKPtoRAMCKP/benchmark/instance_5_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_5_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_10_3_',
                'CCMCKPtoRAMCKP/benchmark/instance_10_5_',
                'CCMCKPtoRAMCKP/benchmark/instance_10_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_10_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_20_3_',
                'CCMCKPtoRAMCKP/benchmark/instance_20_5_',
                'CCMCKPtoRAMCKP/benchmark/instance_20_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_20_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_30_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_30_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_50_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_50_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_100_10_',
                'CCMCKPtoRAMCKP/benchmark/instance_100_20_',
                'CCMCKPtoRAMCKP/benchmark/instance_100_50_']  

        Tmax = [\
                [6210.8,6511.6,7414],
                [4919.9,5404.8,6859.5],
                [3646.8,4255.6,6082],
                [3812.9,4468.8,6436.5],
                [8315.3,8875.6,10556.5],
                [7629.6,8455.2,10932],
                [6743.2,7742.4,10740],
                [6070.4,7137.8,10340],
                [19094.4,20135,23260],
                [16431.4,17953,22521],
                [13278.5,15133,20696.5],
                [11807.6,14023,20670],
                [33404.2,35975,43689],
                [30608,33709,43012],
                [26362,30172,41602],
                [24027.1,28366,41383.5],
                [37851.2,43617,60916],
                [36520.4,42890,62002],
                [63168.1,72851,101900.5],
                [58517.1,69339,101805.5],
                [122523.6,141964,200286],
                [114579.3,136463,202116.5],
                [110990.4,133984,202968]]

        for i in range(len(instance_folder)):
            param = re.findall(r'\d+',instance_folder[i])
            if i < 8:
                ps = populationsize[0]
                evastop = maxiter_[0]    
            elif i < 18:
                ps = populationsize[1]
                evastop = maxiter_[1]
            else:
                ps = populationsize[2]
                evastop = maxiter_[2]

            for j in range(len(Tmax[i])):
                T_m = Tmax[i][j]
                file_ = f'{algo}_{int(param[0])}_{int(param[1])}_{T_m}.csv'
                create_csv(file_, algo)

                for k in range(Iterative_time):                
                    p = _Process(target=main_exp,args=(ps, pc, es, evastop, k, T_m, CL, param, algo, instance_folder[i], file_))
                    p.start()
                    process_list.append(p)
        
                    while len(process_list)>=30:
                        for p in process_list:
                            if not p.is_alive():
                                p.join()
                                process_list.remove(p)
                        if len(process_list)>=30:
                            time.sleep(1)

    for p in process_list:
        p.join()

    print(f"{algo} 实验已结束！")