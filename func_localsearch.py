import re
import time
import heapq
import random
import numpy as np
from random import choice
from solution_operation import solution_initialization, solution_evaluation, solution_cost
from item_operation import Get_item

def Construct_Procedure(Fw, Fu, item2obj, CL, Tmax, param, la, ra):

    Eval_times = 0
    Solution = solution_initialization(Fu, param, 0)
    _St = np.copy(Solution)
    available_S = np.copy(Solution)
    _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
    stopping_condition = -np.ones(len(available_S),int)
    select_factor = -1
    Eval_times += 1

    while 1:
        item_list = []
        for item_id in _St:
            item_list.append(item2obj[item_id])

        solution_weight = [item.weight for item in item_list]

        for i in range(len(available_S)):
            if available_S[i] == -1:
                solution_weight[i] = -1

        select_factor = np.argsort(solution_weight)[-1] # 当前解中weight最大的item所在的factor
        select_item = _St[select_factor]

        # 找到weight矩阵中比item小的下一个因子，对应到utility矩阵中的序号
        if Fw[select_factor].index(select_item) < len(Fw[select_factor])-1: 

            _St[select_factor] = Fw[select_factor][Fw[select_factor].index(select_item)+1]
            _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
            c_ = solution_cost(_St, item2obj)
            # print("当前解：",c_ ,_pt,_St)
            Eval_times += 1
            if _pt >= CL:
                # print("Yeah!")
                break
        else:
            # 随后的循环不再替换本次的select_factor
            available_S[select_factor] = -1
            if (available_S == stopping_condition).all():
                print("Warning: 没有找到可行解, 初始解设为每个class下最小weight的item!")
                _St = solution_initialization(Fw, param, -1)
                break   

    init_feasible_Solution = _St
    init_obj_cost = solution_cost(_St, item2obj)
    init_p = _pt

    return init_feasible_Solution, init_obj_cost, init_p, Eval_times

def localsearch(Fw, Fu, Fv, item2obj, Tmax, CL, param, maxiter, la, ra):

    _St, _ct, _pt, Eval_times = Construct_Procedure(Fw, Fu, item2obj, CL, Tmax, param, la, ra)

    S_opt = np.copy(_St)
    c_opt = _ct
    _p = _pt
    t = 0

    while t <= maxiter:
        ex_factor = []
        ex_item = [item for item in _St]
        _ct, _pt, _St, eva = Degrade_(_St, _pt, Fv, item2obj, Tmax, CL, param, ex_factor, ex_item, la, ra)
        Eval_times += eva

        _ct,_pt,_St,Eval_times = Local_Swap_Search(_St, _pt, Fv, item2obj, Tmax, CL, param, Eval_times, la, ra)
        
        if _ct < c_opt:
            S_opt = np.copy(_St)
            c_opt = _ct
            _p = _pt

        t += 1
        

    c_opt, _p, S_opt, Eval_times = Further_Swap_Search(S_opt, _p, Fv, item2obj, Tmax, CL, param, Eval_times, la, ra)
       
    Optimal_Solution = S_opt
    Optimal_cost = c_opt
    Optimal_p = _p

    return   Optimal_Solution, Optimal_cost, Optimal_p,Eval_times

def Local_Swap_Search(Solution, p, Fv, item2obj, Tmax, CL, param, Eval_times, la, ra):
    factor_list = [i for i in range(int(param[0]))]
    _St,_S = np.copy(Solution), np.copy(Solution)
    _p = p
    random.shuffle(factor_list)

    for factor_id in factor_list:
        select_item = _St[factor_id]
        # 遍历value矩阵中比item大的所有因子
        index_ = Fv[factor_id].index(select_item)

        for item_id in Fv[factor_id][:index_][::-1]:
            _St[factor_id] = item_id
            _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
            Eval_times += 1
            if _pt >= CL:
                _S = np.copy(_St)   # 尽可能找到最大value且满足置信度约束的那个因子
                _p = _pt
                _c = solution_cost(_S, item2obj)
                break               # factor_resort_v从大到小排列，找到的第一个可行item即可跳出    
        _St = np.copy(_S)   # 更新_St

    feasible_Solution = _S
    obj_cost = solution_cost(_S, item2obj)
    p = _p
    return  obj_cost, p, feasible_Solution, Eval_times

def Degrade_(Solution, _pp, Fw, item2obj, Tmax, CL, param, ex_factor, ex_item, la, ra):

    _St = np.copy(Solution)
    Eval_times = 0

    factor_id = choice([i for i in range(int(param[0])) if i not in ex_factor]) # 随机选择一个禁忌集外的factor
    factor_item_ex =[_St[factor_id]]

    _St[factor_id] = choice([item for item in Fw[factor_id] if item not in ex_item])   # 随机选择一个非原因子的因子  
    factor_item_ex.append(_St[factor_id]) # 更新当前factor的禁忌集
    _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
    Eval_times += 1

    while _pt < CL:

        if len(factor_item_ex) < len(Fw[factor_id]):

            _St[factor_id] = choice([item for item in Fw[factor_id] if item not in factor_item_ex])   # 随机选择一个非原因子的因子  
            factor_item_ex.append(_St[factor_id]) # 更新当前factor的禁忌集
            _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
            Eval_times += 1

        elif len(ex_factor)<int(param[0])-1:
            # 该factor除了原item外所有其他item都不可行，恢复原始解，并更换factor
            ex_factor.append(factor_id) # 更新factor的禁忌集        
            _, _pt, _St, eval= Degrade_(Solution, _pp, Fw, item2obj, Tmax, CL, param, ex_factor, ex_item, la, ra)
            Eval_times = eval

        else:
            # print("当前输入不能做任何的degrade!!!\n")
            _pt = _pp
            return solution_cost(Solution, item2obj),_pt,Solution,Eval_times

    # print(f"The Evaluation times of Degrade_ procedure is {Eval_times}")
    return solution_cost(_St, item2obj), _pt,_St, Eval_times

def Further_Swap_Search(Solution, p, Fv, item2obj, Tmax, CL, param, Eval_times, la, ra):

    _St = np.copy(Solution)
    _c = solution_cost(_St, item2obj)
    _p = p
    factor_list = [i for i in range(int(param[0]))]
    random.shuffle(factor_list)
    
    # 遍历所有factor对
    for i in range(int(param[0])-1):
        factor_id1 = factor_list[i]
        for j in range(i + 1,int(param[0])):
            factor_id2 = factor_list[j]
            solution_group = []
            # 遍历value矩阵中比原value大的因子组合
            for item_id1 in Fv[factor_id1]:
                for item_id2 in Fv[factor_id2]:
                    S_temp = np.copy(_St)
                    S_temp[factor_id1], S_temp[factor_id2]= item_id1, item_id2
                    c_st = solution_cost(S_temp, item2obj)
                    if c_st < _c:
                        # 构造堆
                        heapq.heappush(solution_group,list([c_st,S_temp]))

            while solution_group: 
                # 默认弹出最小值，因为取负，所以是比当前解cost小的最大值
                temp = heapq.heappop(solution_group)
                _pt = solution_evaluation(temp[1], item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
                Eval_times += 1
                if _pt >= CL:
                    _St = np.copy(temp[1])   # 更新最优解
                    _p = _pt
                    _c = temp[0]
                    break       # 不再继续评估当前class组合，开始下一组class

    feasible_Solution = np.copy(_St)
    obj_cost = solution_cost(_St, item2obj)
    p = _p
    # print(f"The Evaluation times of further swap search procedure is {Eval_times}")

    return  obj_cost, p, feasible_Solution, Eval_times