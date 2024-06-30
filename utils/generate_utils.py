import random
import math
from scipy import stats
import numpy as np
from functools import partial

# path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ppgraph import PPNode


def get_funct_for_attr(rule_name):
    functions = {
        'name': _generate_new_name,
        'branch_num': _generate_branch_num,
        'storage': _generate_storage,
        'demands': _generate_demands,
        'orders': _generate_orders,
        'ratio': _calc_ratio,
    }
    assert rule_name in functions.keys()
    return partial(functions[rule_name])


def _calc_ratio(max_depth):
    """
    calculate sub_ratio and cross_ratio by depth (base * decay ^ depth)
    """
    # begin from 3rd layer (apart from root and product node)
    sub_ratio_origin = 0.4  # 0.5  # possibility of a node with subs
    sub_ratio_decay = 0.85  # 0.9
    cross_ratio_origin = 0.6  # 0.8  # possibility of s_node with crossing for sub (sub_ratio * cross_ratio)
    cross_ratio_decay = 0.85  # 0.9
    sub_ratio = [sub_ratio_origin]
    cross_ratio = [cross_ratio_origin]

    for i in range(0, max_depth + 1):
        sub_ratio.append(sub_ratio[i] * sub_ratio_decay)
        cross_ratio.append(cross_ratio[i] * cross_ratio_decay)
    return sub_ratio, cross_ratio


def _generate_new_name(node_type, name_counts):
    if node_type == 'material':
        name = 'material{}'.format(name_counts['material'])
        name_counts['material'] += 1
    elif node_type == 'group':
        name = 'group{}'.format(name_counts['group'])
        name_counts['group'] += 1
    elif node_type == 'order':
        name = 'order{}'.format(name_counts['order'])
        name_counts['order'] += 1
    else:
        raise NotImplementedError
    return name


def _generate_branch_num(cross_ratio, sign, depth):
    """
    sign: parents' node_type; 'p' for products

    get number of branches for a node
    """
    cross_rate = cross_ratio[depth]

    # get total branch num
    # 'p'
    if sign == 'material' and depth == -1:
        # uniform
        total_branch_num = random.randint(0, 8)
    # 's'
    elif sign == 'group':
        # uniform
        total_branch_num = random.randint(2, 6)
    # 'm'
    elif sign == 'material':
        # uniform
        total_branch_num = random.randint(0, 5)
    else:
        raise NotImplementedError

    # get cross branch num
    cross_branch_num = math.ceil(total_branch_num * cross_rate)
    new_branch_num = total_branch_num - cross_branch_num

    return new_branch_num, cross_branch_num


def _generate_storage(sign):
    """

    get storage for a node:
    1. get node type(product, substitute, material, infinite)
    2. get storage according to node type

    return int
    """

    # get node type
    prob = [0.45, 0.19, 0.27, 0.09]
    if sign == 'product':
        storage_type = 'product'
    else:
        storage_type = np.random.choice(['product', 'substitute_group', 'material', 'infinite'], p=prob)

    if storage_type == 'product':
        # gamma distribution
        data = np.round(stats.gamma.rvs(a=0.335, loc=1.0, scale=112, size=1))[0] + random.randint(0, 100)
    elif storage_type == 'substitute_group':
        # gamma distribution
        data = np.round(stats.gamma.rvs(a=0.000234, loc=0.0, scale=4.268, size=1))[0] + random.randint(0, 100)
    elif storage_type == 'material':
        # gamma distribution
        data = np.round(stats.gamma.rvs(a=0.0028, loc=0.0, scale=4.061, size=1))[0] + random.randint(0, 100)
    elif storage_type == 'infinite':
        # given number
        data = 9000000
    else:
        raise NotImplementedError

    return data


def _generate_demands(num):
    prob_lo_hi = [0.98, 0.02]

    # decide whether to generate low or high demand
    lo_hi = np.random.choice(['lo', 'hi'], p=prob_lo_hi, size=num)
    lo_num = np.sum(lo_hi == 'lo')
    hi_num = np.sum(lo_hi == 'hi')

    # get demand
    lo_demands, hi_demands = [], []
    # lo
    if lo_num:
        # gamma
        lo_demands = np.clip(np.round(stats.gamma.rvs(a=0.392, loc=1, scale=103.14, size=lo_num)), a_min=1, a_max=1e5).tolist()
    # hi
    if hi_num:
        # rayleigh
        hi_demands = np.clip(np.round(stats.rayleigh.rvs(loc=-1211.55, scale=2083.56, size=hi_num)), a_min=1, a_max=1e5).tolist()

    # concatenate
    demands = np.concatenate((lo_demands, hi_demands), axis=0)

    return demands


def _generate_orders(order_num, name_counts):
    """
    generate all orders
    """
    # prob for each type of order: normal, shared
    prob = [0.19, 0.81]

    # generate orders
    # get num for each type of order (normal, shared)
    order_num = int(order_num * min(max(1 + random.random(), 0.8), 1.2))
    order_types = np.random.multinomial(order_num, prob, size=1)[0]
    normal_num, shared_group_num = order_types[0], order_types[1]
    if normal_num == 0:
        normal_num = 1
        shared_group_num -= 1
    elif shared_group_num == 0:
        normal_num -= 1
        shared_group_num = 1

    # get 'how many orders share the same product' for each group of shared order
    shared_num_by_group = np.random.normal(loc=3, scale=5, size=shared_group_num)
    # round
    shared_num_by_group = np.round(shared_num_by_group)
    # clip
    shared_num_by_group = np.clip(np.abs(shared_num_by_group), 1, 15)
    # convert to int
    shared_num_by_group = shared_num_by_group.astype(int)

    # get demands for each type of order
    normal_demands = _generate_demands(normal_num)
    shared_demands = _generate_demands(sum(shared_num_by_group))

    # randomly generate priority for each order
    priorities = list(np.arange(0, normal_num + len(shared_demands)))
    random.shuffle(priorities)

    ## generate orders
    orders = []
    # normal
    for i in range(normal_num):
        order_node = PPNode(
            id=_generate_new_name('order', name_counts), 
            node_type='order', 
            demand=normal_demands[i]
        )
        order_node.priority = priorities[i]
        orders.append([order_node])

    # shared
    start = 0
    for i in range(shared_group_num):
        # get demands for this group
        all_demands = shared_demands[start:start + shared_num_by_group[i]]
        all_priorities = priorities[start+normal_num: start+normal_num+shared_num_by_group[i]]
        start += shared_num_by_group[i]
        order_group = []
        # get priority for this group
        # generate orders
        for j in range(shared_num_by_group[i]):
            order_node = PPNode(
                id=_generate_new_name('order', name_counts), 
                node_type='order', 
                demand=all_demands[j]
            )
            order_node.priority = all_priorities[j]
            order_group.append(order_node)
        orders.append(order_group)

    return orders



