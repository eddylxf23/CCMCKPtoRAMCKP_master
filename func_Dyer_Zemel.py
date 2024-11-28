import sys
sys.path.append('.')

from item_operation import Get_weight
import copy

class Good():
    def __init__(self, row, col, p, w):
        self.row = row
        self.col = col
        self.p = p
        self.w = w
    def __repr__(self):
        return str((self.w, self.p))
 
def goods_filter_LP_dominated(goods, exacting=True):
    [goods[i].sort(key=lambda n:n.w, reverse=False) for i in range(len(goods))]
    goods1 = [[goods[i][0]] for i in range(len(goods))]

    for i in range(len(goods)):
        p, w = goods1[i][0].p, goods1[i][0].w
        for j in range(1, len(goods[i])):
            if goods[i][j].w == w and goods[i][j].p > p:
                goods1[i][-1] = goods[i][j]
                p = goods[i][j].p
            elif goods[i][j].w > w and goods[i][j].p > p:
                goods1[i].append(goods[i][j])
                p, w = goods[i][j].p, goods[i][j].w
    # 严格LP_dominated
    if exacting:
        goods2 = [[goods1[i][0]] for i in range(len(goods1))]
        for i in range(len(goods1)):
            j = 0
            while j < len(goods1[i]) - 1:
                slopes = [[k, (goods1[i][k].p - goods1[i][j].p) / (goods1[i][k].w - goods1[i][j].w)] for k in range(j + 1, len(goods1[i]))]
                slopes.sort(key=lambda x: x[1], reverse=True)
                j = slopes[0][0]
                goods2[i].append(goods1[i][j])
        goods1 = goods2
 
    return goods1
 
 
#每次更新都选择局部提升最大的
def Dynamic(W, P, n2f, c):

    goods = [[Good(i, j, P[i][j], W[i][j]) for j in range(len(W[i]))] for i in range(len(W)) if len(W[i]) > 0]
    G = copy.deepcopy(goods)
    goods = goods_filter_LP_dominated(G, exacting=False)
 
    class Combination():
        def __init__(self, P, W, L):
            self.P = P
            self.W = W
            self.L = L
 
        def __repr__(self):
            return str((self.W, self.P))
 
    C = {goods[0][j].w: Combination(goods[0][j].p, goods[0][j].w, [goods[0][j]]) for j in range(len(goods[0]))}
 
    for i in range(1, len(goods)):
        C_next = {}
        for j in range(len(goods[i])):
            for k,v in C.items():
                if goods[i][j].w + k > c:
                    continue
                #不在或价值更高则更新
                if (goods[i][j].w + k not in C_next) or (goods[i][j].w + k in C_next and goods[i][j].p + v.P > C_next[goods[i][j].w + k].P):
                    C_next[goods[i][j].w + k] = Combination(goods[i][j].p + v.P, goods[i][j].w + k, v.L+[goods[i][j]])
        C = C_next
 
    if len(C) != 0:
        res = list(C.values())
        res.sort(key=lambda x:x.P, reverse=True)
        res_choose = res[0].L
        obj_value = res[0].P
        weight = res[0].W
        solution = []

        for i in range(len(goods)):
            column = res_choose[i].col
            ind_ = n2f[i][column].id
            solution.append(ind_)

        return solution
    
    else:
        return None


def dp(Value, f2i, lambda_itera, Tmax_itera):

    W = Get_weight(f2i, lambda_itera)

    return Dynamic(W, Value, f2i, Tmax_itera)


if __name__ == '__main__':

    Tmax = 14
    CL = 0.9