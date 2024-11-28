import re
import time
import heapq
import random
import numpy as np
from item_operation import Get_item
from solution_operation import solution_initialization, solution_evaluation, solution_cost
from func_localsearch import Local_Swap_Search, Further_Swap_Search

# 种群初始化
def PopulationInitialization(PopulationSize, Fw, Tmax, item2obj, CL, param, eva, la, ra):
    factor_num = int(param[0])
    InitPopulation = [] 
    for _ in range(PopulationSize):
        Single = []
        for factor_index in range(factor_num):
            Single.append(random.choice(Fw[factor_index]))
        _cost, _p, SingleRepair, e = RepairSolution(Single, Fw, Tmax, item2obj, CL, param, la, ra)
        SingleRepair.insert(0,-_p)
        SingleRepair.insert(1,_cost)  
        heapq.heappush(InitPopulation,SingleRepair)
        eva += e
    return InitPopulation, eva

# 交叉算子
def CrossOver(pc,ParentA,ParentB,param):
    factor_num = int(param[0])
    Child = [-1 for _ in range(factor_num)]
    Child[0] = ParentA[2]
    sign = 1
    for factor_index in range(factor_num):
        x = random.uniform(0,1)
        if x < pc:
            sign = sign*-1  # 如果满足条件，则对位点进行交叉

        if sign == 1:
            Child[factor_index] = ParentA[factor_index + 2]
        else:
            Child[factor_index] = ParentB[factor_index + 2]
    return Child

# 变异算子
def Mutation(Single, param, Fw):
    factor_num = int(param[0])
    pm = 1/factor_num
    for factor_index in range(factor_num):
        factor_list = Fw[factor_index].copy()
        factor_list.remove(Single[factor_index])
        x = random.uniform(0,1)
        if x < pm:
            Single[factor_index] = random.choice(factor_list) # 如果满足条件，则对当前位点进行随机变异（除原因子）
    return Single

# 按概率挑选个体
def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break    
    return item, item_probability

# 优势个体挑选
def ParentPick(Population, Value):
    Probability = list(Value/sum(Value))
    ParentA, probA = random_pick(Population, Probability)    # 按每个环节内的概率向量随机选择个体A

    Population.remove(ParentA)
    Aindex = Probability.index(probA)
    Value.remove(Value[Aindex])
    Probability = Value/sum(Value)
    ParentB, _ = random_pick(Population, Probability)   # 按每个环节内的概率向量随机选择个体B
    return ParentA, ParentB

# 修复模块
def RepairSolution(Solution, Fw, Tmax, item2obj, CL, param, la, ra):
    _St = np.copy(Solution)
    available_S = np.copy(Solution)
    Eval = 0
    stopping_condition = -np.ones(len(available_S),int)
    select_factor = -1
    _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
    Eval += 1

    if _pt < CL:
        while 1:
            factor_list = []
            for item_id in _St:
                factor_list.append(item2obj[item_id])

            solution_weight = [item.weight for item in factor_list]
            for i in range(len(available_S)):
                if available_S[i] == -1:
                    solution_weight[i] = -1

            select_factor = np.argsort(solution_weight)[-1] # 当前解中weight最大的item所在的factor
            select_item = _St[select_factor]
    
            if Fw[select_factor].index(select_item) < len(Fw[select_factor])-1:     # 查询weight矩阵中比当前item小的下一个因子
                _St[select_factor] = Fw[select_factor][Fw[select_factor].index(select_item)+1]          
                _pt = solution_evaluation(_St, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
                Eval += 1
                if _pt >= CL:
                    break
            else:  
                available_S[select_factor] = -1   # 如果不再有更小的weight，随后的循环不再替换本次的select_factor
                if (available_S == stopping_condition).all():
                    # print("Warning: 没有找到可行解, 设为每个class下最小weight的item!")
                    _St = solution_initialization(Fw, param, -1)
                    break   

    return solution_cost(_St,item2obj), _pt, list(np.copy(_St)), Eval

# GA主函数
def GeneticAlgorithm(ps, pc, es, evastop, Fw, Fv, Tmax, item2obj, CL, param, la, ra, localsearch_sign = None):
    #----------- 初始化confi_array -----------
    eval_times = 0
    eva = 0

    SolutionPopulation, eval_times = PopulationInitialization(ps, Fw, Tmax, item2obj, CL, param, eval_times, la, ra)

    while eva < evastop:
        #========== 选取优势个体作为父母种群 =============
        ElitePopulation = []
        EliteValue = []
        ElitePopulation = heapq.nsmallest(es,SolutionPopulation)
        EliteValue = [1/(EP[1]+100*(1-EP[0])) for EP in ElitePopulation]
        # EliteValue = [1/(EP[0]) for EP in ElitePopulation]

        for _ in range(int(es/2)):
            ParentA, ParentB = ParentPick(ElitePopulation,EliteValue)
            #========== 交叉，变异，修复 =============
            Child = CrossOver(pc, ParentA, ParentB, param)
            Child = Mutation(Child,param,Fw)
            cost,pt,Child,et = RepairSolution(Child, Fw, Tmax, item2obj, CL, param, la, ra)
            # cost = solution_cost(Child,item2obj)
            # pt = solution_evaluation(Child, item2obj, Tmax, CL, _lambda = la, risk_averse = ra)
            # et = 1
            eval_times += et
            if localsearch_sign is not None:
                cost,pt,Child,eval_times = Local_Swap_Search(Child, pt, Fv, item2obj, Tmax, CL, param, eval_times)
                Child = list(Child)
            
            Child.insert(0,-pt)
            Child.insert(1,cost)
            
            heapq.heappush(SolutionPopulation,Child)

        SolutionPopulation = heapq.nsmallest(ps,SolutionPopulation)
        heapq.heapify(SolutionPopulation)
        # print(eva)
        eva += 1

    St = heapq.heappop(SolutionPopulation)
    # Optimal_cost, Optimal_p,  Optimal_Solution, et = RepairSolution(St[2:], Fw, Tmax, item2obj, CL, param, la, ra)
    Optimal_Solution = np.copy(St[2:])
    Optimal_p = -St[0]
    Optimal_cost = St[1]

    if localsearch_sign is not None:
        Optimal_cost, Optimal_p, Optimal_Solution, eval_times = Further_Swap_Search(Optimal_Solution, Optimal_p, Fv, item2obj, Tmax, CL, param, eval_times, la, ra)
    
    return  Optimal_Solution, Optimal_cost, Optimal_p, eval_times

if __name__ == '__main__':
    folder_name = 'CCMCKPtoRAMCKP/benchmark/instance_5_5_'
    Tmax = 7629.6
    CL = 0.99
    param = re.findall(r'\d+',folder_name)
    maxiter_ = 200
    populationsize = 200
    pc = 0.6
    es = 0.6
    Value, f2i, item2obj, Fw, Fu, Fv = Get_item(param, 0, folder_name)
    Elitesize = int(populationsize*es) 
    _lambda = 0
    ti = time.time()
    optimal_solution, optimal_obj, confilevel, eval_times =  GeneticAlgorithm(populationsize, pc, Elitesize, maxiter_ , Fw, Fv, Tmax, item2obj, CL, param, _lambda, ra =None)
    runtime = time.time() - ti
    print(f"\nPure GA 最优解为:", optimal_solution, "最优成本为:", optimal_obj, "运行时间为:", runtime, "评估结果：", confilevel, "Evaluation次数:", eval_times)  
    print(solution_evaluation(optimal_solution, item2obj, Tmax, CL, MonteCarlo = 1*10**7))
