import os
import sys
from utils import MKNode, MKGraph, build_mip
import random
from collections import deque
from tqdm import tqdm
import math
import numpy as np
from scipy import stats


class RandomGenerator:
    def __init__(self, output_dir, seed, instance_num, max_depth, products_num, order_num,
                 mip_solving_time_ub=300):
        # settings
        self.output_dir = output_dir
        self.instance_num = instance_num
        self.visual = False
        self.visual_with_data = False
        self.fixed_order = False
        self.need_simplification = True

        # mip
        self.mip_solver = 'scip'
        self.obj_type = 'product_num'
        self.mip_solving_time_ub = mip_solving_time_ub

        # problem_size
        self.max_depth = max_depth  # height of separated product tree
        self.products_num = products_num  # num of products
        self.order_num = order_num  # num of orders

        # set sub_ratio and cross_ratio: probability of sub and cross (depth 0 is children of product)
        self.sub_ratio = []
        self.cross_ratio = []
        self.calc_ratio()

        # init
        self.root = None
        self.name_count_n = 0
        self.name_count_v = 0
        self.name_count_o = 0
        random.seed(seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_branch_num(self, sign, depth):
        """
        sign: parents' node_type; 'p' for products

        get number of branches for a node
        """
        cross_rate = self.cross_ratio[depth]

        # get total branch num
        # 'p'
        if sign == 'm' and depth == -1:
            # uniform
            total_branch_num = random.randint(0, 8)
        # 'v'
        elif sign == 'v':
            # uniform
            total_branch_num = random.randint(2, 6)
        # 'm'
        elif sign == 'm':
            # uniform
            total_branch_num = random.randint(0, 5)
        else:
            raise NotImplementedError

        # get cross branch num
        cross_branch_num = math.ceil(total_branch_num * cross_rate)
        new_branch_num = total_branch_num - cross_branch_num

        return new_branch_num, cross_branch_num

    def get_storage(self, sign):
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
            storage_type = np.random.choice(['product', 'substitute', 'material', 'infinite'], p=prob)

        if storage_type == 'product':
            # gamma distribution
            data = np.round(stats.gamma.rvs(a=0.335, loc=1.0, scale=112, size=1))[0] + random.randint(0, 100)
        elif storage_type == 'substitute':
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

    def get_orders(self):
        """
        generate all orders
        """

        def generate_demands(num):
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
                lo_demands = np.clip(np.round(stats.gamma.rvs(a=0.392, loc=1, scale=103.14, size=lo_num)), a_min=1,
                                     a_max=1e5).tolist()
            # hi
            if hi_num:
                # rayleigh
                hi_demands = np.clip(np.round(stats.rayleigh.rvs(loc=-1211.55, scale=2083.56, size=hi_num)), a_min=1,
                                     a_max=1e5).tolist()

            # concatenate
            demands = np.concatenate((lo_demands, hi_demands), axis=0)

            return demands

        # prob for each type of order: normal, shared
        prob = [0.19, 0.81]

        # generate orders
        # get num for each type of order (normal, shared)
        order_num = int(self.order_num * min(max(1 + random.random(), 0.8), 1.2))
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
        normal_demands = generate_demands(normal_num)
        shared_demands = generate_demands(sum(shared_num_by_group))

        # generate orders
        orders = []
        # normal
        for i in range(normal_num):
            order_node = MKNode(name=self.get_new_name('o'), node_type='o', v=normal_demands[i])
            orders.append([order_node])

        # shared
        start = 0
        for i in range(shared_group_num):
            # get demands for this group
            all_demands = shared_demands[start:start + shared_num_by_group[i]]
            start += shared_num_by_group[i]
            order_group = []
            # get priority for this group
            # generate orders
            for j in range(shared_num_by_group[i]):
                order_node = MKNode(name=self.get_new_name('o'), node_type='o', v=all_demands[j])
                order_group.append(order_node)
            orders.append(order_group)

        return orders

    def calc_ratio(self):
        """
        calculate sub_ratio and cross_ratio by depth (base * decay ^ depth)
        """
        # begin from 3rd layer (apart from root and product node)
        sub_ratio_origin = 0.4
        sub_ratio_decay = 0.85
        cross_ratio_origin = 0.6
        cross_ratio_decay = 0.85
        self.sub_ratio = [sub_ratio_origin]
        self.cross_ratio = [cross_ratio_origin]

        for i in range(0, self.max_depth + 1):
            self.sub_ratio.append(self.sub_ratio[i] * sub_ratio_decay)
            self.cross_ratio.append(self.cross_ratio[i] * cross_ratio_decay)

    def reset(self, graph):
        # reset
        self.root = None
        self.name_count_n = 0
        self.name_count_v = 0
        self.name_count_o = 0

    def run(self):
        # generate
        for instance_idx in tqdm(range(1, self.instance_num + 1), file=sys.stdout, desc='data generating'):
            # build tree with separated products
            self.root = MKNode(name='root', node_type='r', v=0, depth=0)

            # map by depth; for nodes whose parent is material and virtual node respectively
            node_by_depth_m = [[] for _ in range(self.max_depth + 1)]
            node_by_depth_v = [[] for _ in range(self.max_depth + 1)]

            # build graph (given product num)
            product_num = int(self.products_num * min(max(1 + random.random(), 0.8), 1.2))
            for i in range(product_num):
                node_by_depth_m_copy = [[exist_v_node for exist_v_node in map_at_depth]
                                        for map_at_depth in node_by_depth_m]
                node_by_depth_v_copy = [[exist_v_node for exist_v_node in map_at_depth]
                                        for map_at_depth in node_by_depth_v]

                # build branch
                product_storage = self.get_storage(sign='product')  # NOTICE: depth=-1 for node_by_depth_copy
                product_node = MKNode(self.get_new_name('m'), node_type='m', v=product_storage, depth=-1)
                self.build_branch(root=product_node,
                                  node_by_depth_total=(node_by_depth_m_copy, node_by_depth_v_copy))
                # save
                self.root.add_child(product_node)
                product_node.add_parent(self.root)
                node_by_depth_m = node_by_depth_m_copy
                node_by_depth_v = node_by_depth_v_copy

            # build orders
            # form order node: only normal materials (not substitutes) are considered
            self.build_orders(node_by_depth=node_by_depth_m)

            # build graph
            generated_graph = MKGraph(root=self.root)

            # build and output mip
            save_dir = os.path.join(self.output_dir, 'instance_{}.mps'.format(instance_idx))
            build_mip(graph=generated_graph, save_dir=save_dir)

            # output stat
            tqdm.write('instance {} saved'.format(instance_idx))

            # reset
            self.reset(generated_graph)
        print('Generation finished. Data saved to:', self.output_dir)

    def build_branch(self, root, node_by_depth_total):
        # build branch for a single product
        # copy map: cannot cross within one product
        node_by_depth_m, node_by_depth_v = node_by_depth_total
        node_by_depth_m_extended = [[m for m in m_at_depth] for m_at_depth in node_by_depth_m]
        node_by_depth_v_extended = [[m for m in v_at_depth] for v_at_depth in node_by_depth_v]
        # extend list: depth i include i->max_depth
        for i in range(len(node_by_depth_m) - 2, -1, -1):
            node_by_depth_m_extended[i].extend(node_by_depth_m_extended[i + 1])
            node_by_depth_v_extended[i].extend(node_by_depth_v_extended[i + 1])

        # initialize deque
        dq = deque()
        dq.append(root)

        # BFS by layer; control depth: node.depth<max_depth
        while dq:
            node = dq.popleft()
            # create nodes
            # TODO: edit; judge node type: m, v
            new_branch_num, cross_branch_num = self.get_branch_num(sign=node.node_type, depth=node.depth)

            # generate crossed nodes
            # choose 'cross_branch_num' nodes from map (m and v separately)
            if node.node_type == 'm':
                nodes_for_chosen = node_by_depth_m_extended[node.depth + 1]
            else:
                nodes_for_chosen = node_by_depth_v_extended[node.depth + 1]

            # cross: link to node
            if len(nodes_for_chosen) > 0:
                crossed_nodes = random.sample(nodes_for_chosen, min(cross_branch_num, len(nodes_for_chosen)))
                # link
                for node_to_cross in crossed_nodes:
                    # TODO: depth ratio
                    # cross: link to node
                    node.children.append(node_to_cross)
                    node_to_cross.parents.append(node)
            # avoid v-single m
            if node.node_type == 'v' and min(cross_branch_num, len(nodes_for_chosen)) + new_branch_num <= 1:
                new_branch_num = 2

            # generate new nodes
            for _ in range(new_branch_num):
                # virtual node: avoid v-v edges
                if node.node_type != 'v' and random.random() < self.sub_ratio[node.depth + 1]:
                    new_node = MKNode(name=self.get_new_name('v'), node_type='v', v=0, depth=node.depth)
                    # 100% satisfy: depth<max_depth -> extend
                    dq.append(new_node)

                # normal nodes
                else:
                    # sign is not p: whatever its value, result is all the same
                    new_node_storage = self.get_storage(sign=node.node_type)
                    new_node = MKNode(name=self.get_new_name('m'), node_type='m',
                                      v=new_node_storage, depth=node.depth + 1)

                    # add to map
                    if node.node_type == 'm':
                        node_by_depth_m[new_node.depth].append(new_node)
                    else:
                        node_by_depth_v[new_node.depth].append(new_node)

                    # satisfy: depth<max_depth -> extend
                    if new_node.depth < self.max_depth:
                        dq.append(new_node)

                # connect to parent
                new_node.add_parent(node)
                node.add_child(new_node)

    def build_orders(self, node_by_depth):
        """
        build four types of orders:
            1. normal ones: 1 order - 1 product
            2. middle products
            3. complete products
            4. shared products
        """

        # get orders
        orders = self.get_orders()

        # get products: part1-complete; part2-middle
        products_complete = self.root.children
        products_middle = []
        # disconnect from root
        self.root.children = []
        for product in products_complete:
            product.parents.remove(self.root)

        # randomly select part2 from combined node_by_depth(list of list of nodes)
        # check if required
        order_num = len(orders)
        if order_num <= len(products_complete):
            products_complete = products_complete[:order_num]
        else:
            # combine
            node_list = []
            for node_list_at_depth in node_by_depth:
                node_list.extend(node_list_at_depth)
            products_middle = random.sample(node_list, order_num - len(products_complete))

        # concatenate
        products = products_complete + products_middle
        random.shuffle(products)

        # solve in pair
        self.root.children = []
        for order_group, product in zip(orders, products):
            for order in order_group:
                # connect order to root
                self.root.add_child(order)
                order.add_parent(self.root)

                # connect order to product
                order.add_child(product)
                product.add_parent(order)

    def get_new_name(self, node_type):
        if node_type == 'm':
            name = 'm{}'.format(self.name_count_n)
            self.name_count_n += 1
        elif node_type == 'v':
            name = 'v{}'.format(self.name_count_v)
            self.name_count_v += 1
        elif node_type == 'o':
            name = 'o{}'.format(self.name_count_o)
            self.name_count_o += 1
        else:
            raise NotImplementedError
        return name


if __name__ == '__main__':
    generator = RandomGenerator(output_dir='../data/', seed=41, instance_num=1000, products_num=1300, order_num=2000,
                                max_depth=5)
    generator.run()
