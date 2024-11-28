import re
import math
import copy
import time 
import random
import numpy as np
from scipy.stats import norm
from func_Dyer_Zemel import dp
from func_localsearch import localsearch
from func_Genetic import GeneticAlgorithm
from item_operation import Get_item, max_var
from solution_operation import solution_cost, solution_evaluation
# import func_timeout
# from func_timeout import func_set_timeout
 
# @func_set_timeout(3600) # 设置函数最大执行时间

def solve_ramckp(ps, pc, es, evastop, maxiter, param, fn, CL, lambda_itera, Tmax_itera, Tmax, alg):

    Value, f2i, item2obj, Fw, Fu, Fv = Get_item(param, lambda_itera, fn)
    risk_averse = True
    if alg == 'LS':
        solu, _, p, et = localsearch(Fw, Fu,Fv, item2obj, Tmax_itera, CL, param, maxiter, lambda_itera, risk_averse)
    elif alg == 'GA':
        solu, _, p, et = GeneticAlgorithm(ps, pc, es, evastop, Fw, Fv, Tmax_itera, item2obj, CL, param, lambda_itera, risk_averse)
    elif alg == 'LS-GA':
        solu, _, p, et = GeneticAlgorithm(ps, pc, es, evastop, Fw, Fv, Tmax_itera, item2obj, CL, param, lambda_itera, risk_averse, localsearch_sign = 1)
    else:
        solu = dp(Value, f2i, lambda_itera, Tmax_itera)
        et = 0

    p = solution_evaluation(solu, item2obj, Tmax, CL)

    return solu, p, et, item2obj, f2i
    

def CCMCKP_main(ps, pc, es, evastop, mi, Tmax, CL, param, algo, fn):
    # Initalization
    _epsilon = 10**(-5)
    lambda_itera = 0
    Tmax_itera = Tmax
    C =  norm.ppf(CL)
    eval_times = 0
    et = 0
    solve_ra_times = 0

    start_time = time.time()
    # k=1, solve initial RA-MCKP, line 1-3
    solu, p, et, item2obj, f2i = solve_ramckp(ps, pc, es, evastop, mi, param, fn, CL, lambda_itera, Tmax_itera, Tmax, algo)
    maximum_total_variance = max_var(f2i)
    eval_times += et
    solve_ra_times += 1
    feasible_solution_set = []
    
    if p>=0:
        # feasible_solution_set.append(sol)
        print("\nThe initial solution is feasible, thus the search process has finished.")
        endtime = time.time()-start_time
        optimal_obj = solution_cost(solu,item2obj)
        optimal_solution = solu
        confidencelevel = solution_evaluation(solu, item2obj, Tmax, CL, MonteCarlo = 1*10**7)

        return optimal_solution, optimal_obj, confidencelevel[1], endtime, eval_times, solve_ra_times
    # line 4-8
    while p<0:
        _sigma_2 = 0
        for j in range(len(f2i)):
            _sigma_2 += item2obj[solu[j]].variance

        _sigma = math.sqrt(_sigma_2)
        lambda_itera = C/_sigma
        solu, p, et, item2obj, f2i = solve_ramckp(ps, pc, es, evastop, mi, param, fn, CL, lambda_itera, Tmax_itera, Tmax, algo)
        solve_ra_times += 1

        if p>=0:
            feasible_solution_set.append(solu)
        
        eval_times += et

    # line 9-10
    point_a = [C**2/lambda_itera**2, Tmax_itera - C**2/lambda_itera]
    point_b = [maximum_total_variance, Tmax_itera - C*math.sqrt(maximum_total_variance)]
    x_a, y_a = point_a[0],point_a[1]
    x_b, y_b = point_b[0],point_b[1]
    # negative sign is required: mu + \lambda*sigma2 <= T
    lambda_ab = -(y_b-y_a)/(x_b-x_a)
    # print(algo, lambda_ab,-C*1/(math.sqrt(maximum_total_variance)+_sigma))
    T_ab = -lambda_ab*(-x_a) + y_a

    # line 11
    LAMBDA_set = [lambda_ab]
    T_set = [T_ab]
    point_set = [[point_a,point_b]]
    LAMBDA_set_tem = []
    T_set_tem = []
    point_set_tem = []
    iteration_times = 0
    # line 12-24
    while len(LAMBDA_set) !=0 :
        iteration_times += 1
        # line 13
        for i in range(len(LAMBDA_set)):
            lambda_itera = LAMBDA_set[i]
            Tmax_itera = T_set[i]

            solu, p, et, item2obj, f2i = solve_ramckp(ps, pc, es, evastop, mi, param, fn, CL, lambda_itera, Tmax_itera, Tmax, algo)
            
            eval_times += et
            solve_ra_times += 1

            if p>=0:
                if any(np.array_equal(solu, arr) for arr in feasible_solution_set):
                    feasible_solution_set.append(solu)
            else:  
                _sigma_2 = 0
                _mu = 0
                # Take out the two corresponding endpoints for calculating this point
                point_1 = point_set[i][0]
                point_2 = point_set[i][1]

                for j in range(len(f2i)):
                    # calculate the sum of mean and the sum of variance in the variance-mean plane
                    _sigma_2 += item2obj[solu[j]].variance
                    _mu += item2obj[solu[j]].mean

                # New infeasible solution coordinate in the variance-mean plane
                point_tem = [_sigma_2, _mu]
                # Calculate the slope and intercept of the line decided by point "tem" and point 1 
                lambda_1 = -((point_tem[1]-point_1[1])/(point_tem[0]-point_1[0]) - _epsilon)
                T_1 = -lambda_1*(-point_1[0]) + point_1[1]
                # Calculate the slope and intercept of the line decided by point "tem" and point 2 
                lambda_2 = -((point_tem[1]-point_2[1])/(point_tem[0]-point_2[0]) + _epsilon)
                T_2 = -lambda_2*(-point_2[0]) + point_2[1]

                LAMBDA_set_tem.append(lambda_1)
                LAMBDA_set_tem.append(lambda_2)
                T_set_tem.append(T_1)
                T_set_tem.append(T_2) 
                point_set_tem.append([point_1, point_tem])
                point_set_tem.append([point_tem, point_2])

        LAMBDA_set = copy.deepcopy(LAMBDA_set_tem)
        T_set = copy.deepcopy(T_set_tem)
        point_set = copy.deepcopy(point_set_tem)
        LAMBDA_set_tem = []
        T_set_tem = []
        point_set_tem = []

        
    obj_list = [solution_cost(s,item2obj) for s in feasible_solution_set]
    if not obj_list:
        return None, [], time.time()-start_time

    endtime = time.time()-start_time
    optimal_obj = min(obj_list)
    optimal_solution = feasible_solution_set[obj_list.index(optimal_obj)]
    confidencelevel = solution_evaluation(optimal_solution, item2obj, Tmax, CL, MonteCarlo = 1*10**7)

    return optimal_solution, optimal_obj, confidencelevel[1], endtime, eval_times, solve_ra_times

if __name__ == '__main__':
    
    # try:
        se = random.seed(233)
        folder_name = folder_name = 'CCMCKPtoRAMCKP/benchmark/instance_5_5_'
        param = re.findall(r'\d+',folder_name)  
        CL = 0.99
        Tmax = 7629.6
        algorithm = ["LS", "DP", "GA", "LS-GA"]
        maxiter_LS = 100

        ps = 500
        pc = 0.6
        es = 0.6
        evastop = 500

        Elitesize = int(ps*es) 

        optimal_solution, optimal_obj, confilevel, runtime, eval_times, solve_ra_times =  CCMCKP_main(ps, pc, Elitesize, evastop, maxiter_LS, Tmax, CL, param, algorithm[0], folder_name)
        print("\nLocal search 最优解为:", optimal_solution, "最优成本为:", optimal_obj, "运行时间为:", runtime, "评估结果：", confilevel, "Evaluation次数:", eval_times, "RA求解次数:", solve_ra_times)

        optimal_solution, optimal_obj, confilevel, runtime, eval_times, solve_ra_times =  CCMCKP_main(ps, pc, Elitesize, evastop, maxiter_LS, Tmax, CL, param, algorithm[2], folder_name)
        print("\nGenetic algorithm 最优解为:", optimal_solution, "最优成本为:", optimal_obj, "运行时间为:", runtime, "评估结果：", confilevel, "Evaluation次数:", eval_times, "RA求解次数:", solve_ra_times)

        optimal_solution, optimal_obj, confilevel, runtime, eval_times, solve_ra_times =  CCMCKP_main(ps, pc, Elitesize, evastop, maxiter_LS, Tmax, CL, param, algorithm[3], folder_name)
        print("\nLS based Genetic algorithm 最优解为:", optimal_solution, "最优成本为:", optimal_obj, "运行时间为:", runtime, "评估结果：", confilevel, "Evaluation次数:", eval_times, "RA求解次数:", solve_ra_times)

        optimal_solution, optimal_obj, confilevel, runtime, _, solve_ra_times =  CCMCKP_main(ps, pc, es, evastop, maxiter_LS, Tmax, CL, param, algorithm[1], folder_name)
        print("\nDynamic programming 最优解为:", optimal_solution, "最优成本为:", optimal_obj, "运行时间为:", runtime, "评估结果：", confilevel, "RA求解次数:", solve_ra_times)

    # except func_timeout.exceptions.FunctionTimedOut:
    #     print("超时了,自动退出")
                    


