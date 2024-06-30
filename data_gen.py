import os
import json
import pickle
import random
from tqdm import tqdm
from collections import deque
import argparse

# path
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.ppgraph import PPGraph, PPNode
# from utils.mip_utils import formulate_mip, solve_mip, MipSolvingConfig

from utils.generate_utils import get_funct_for_attr
from utils.mip_utils import MipSolvingConfig, formulate_mip, solve_mip


class RandomGenerator:
    def __init__(self, output_dir, seed, start_num, end_num, max_depth, product_num, order_num, data_type, time_limit=300):
        # settings
        self.output_config = {
            'output_dir': output_dir,
            'instance_ids': (start_num, end_num),
            'need_mip_result': True,
            'need_simplification': True,
            'data_type': data_type,
        }
        self.id_counters = {
            'material': 0,
            'group': 0,
            'order': 0,
        }
        assert data_type in ['test', 'train']

        # mip
        self.mip_config = {
            'solver_type': 'scip',
            'obj_type': 'product_num',
            'time_limit': time_limit,
            'show_log': False,
        }

        # problem_size
        self.problem_size_config = {
            'max_depth': max_depth,
            'product_num': product_num,
            'order_num': order_num,
        }

        # init
        random.seed(seed)
        # setting output
        self.output_dir = os.path.join(output_dir, 'p{}_o{}'.format(product_num, order_num))
        os.makedirs(self.output_dir, exist_ok=True)

    def reset(self):
        self.id_counters = {
            'material': 0,
            'group': 0,
            'order': 0,
        }


    def run(self):
        # generate
        for instance_idx in tqdm(range(*self.output_config['instance_ids']), file=sys.stdout, desc='data generating'):
            generated_pp_graph = self.generate_graph()
            generated_pp_graph.update_produce_ub()
            tqdm.write(f'graph generated for instance {instance_idx}')
            # generated_pp_graph.display()

            # mip
            if self.output_config['need_mip_result']:
                # test solving
                mip_model = formulate_mip(
                    solving_config=MipSolvingConfig(**self.mip_config),
                    pp_graph=generated_pp_graph
                )
                mip_results = solve_mip(
                    solving_config=MipSolvingConfig(**self.mip_config),
                    model=mip_model, 
                    pp_graph=generated_pp_graph
                )

                # output solve stat to console
                tqdm.write(f'mip solving done within {round(mip_results["solving_time"])} seconds, obj value: {round(mip_results["obj_value"])}')

                # output mip result as json
                mip_result_file_name = os.path.join(self.output_dir, 'mip_result_{}.json'.format(instance_idx))
                with open(mip_result_file_name, 'w') as f:
                    json.dump(mip_results, f)

            # output bom graph object: NOTICE this will delete 'node.parents'
            graph_file_name = os.path.join(self.output_dir, 'graph_{}.pkl'.format(instance_idx))
            generated_pp_graph.save(path=graph_file_name)

            # delete current graph; reset generator
            tqdm.write(f'instance {instance_idx} saved\n')
            generated_pp_graph.delete()
            self.reset()
        print('Generation finished. Data saved to:', self.output_dir)



    def generate_graph(self):
        '''

            Build tree with separated products

        '''
        root = PPNode(id='root', node_type='root')
        
        # map by depth; seperate materials and substitutes
        mapping_by_depth_materials = [[] for _ in range(self.problem_size_config['max_depth'] + 1)]
        mapping_by_depth_substitutes = [[] for _ in range(self.problem_size_config['max_depth'] + 1)]

        # build bom tree (given product num)
        product_num = int(self.problem_size_config['product_num'] * min(max(1 + random.random(), 0.8), 1.2))
        for _ in range(product_num):
            # build sub_root(product) for branch
            product_node = PPNode(
                node_type='material', 
                id=get_funct_for_attr('name')('material', self.id_counters),
                storage=get_funct_for_attr('storage')(sign='product'), 
                depth=-1
            )  # NOTICE: depth=-1 for node_by_depth_copy

            self.build_tree(root=product_node, mapping_by_depth=(mapping_by_depth_materials, mapping_by_depth_substitutes))
            # connect
            PPNode.add_edge(parents=root, children=product_node)
            
        # build orders
        # form order node: only normal materials (not substitutes) are considered
        self.adding_orders(root=root, mapping_by_depth=mapping_by_depth_materials)

        # build graph
        generated_pp_graph = PPGraph(root=root)
        return generated_pp_graph



    def build_tree(self, root, mapping_by_depth):
        '''
            Task: build production-line for a single product
        '''
        sub_ratio, cross_ratio = get_funct_for_attr('ratio')(max_depth=self.problem_size_config['max_depth'])
        # copy map: cannot cross within one product
        node_by_depth_m, node_by_depth_v = mapping_by_depth
        node_by_depth_m_extended = [[m for m in m_at_depth] for m_at_depth in node_by_depth_m]
        node_by_depth_v_extended = [[m for m in v_at_depth] for v_at_depth in node_by_depth_v]
        # extend list: depth i include i->max_depth
        for i in range(len(node_by_depth_m)-2, -1, -1):
            node_by_depth_m_extended[i].extend(node_by_depth_m_extended[i+1])
            node_by_depth_v_extended[i].extend(node_by_depth_v_extended[i+1])

        # initialize deque
        dq = deque()
        dq.append(root)

        # BFS by layer; control depth: node.depth<max_depth
        while dq:
            cur_node = dq.popleft()
            cur_depth = cur_node.depth
            # decide num of nodes to cross and new nodes
            _num_to_branch, _num_to_cross = get_funct_for_attr('branch_num')(cross_ratio=cross_ratio, sign=cur_node.node_type, depth=cur_depth)

            ## crossing: find existing nodes to connect (choose nodes from mapping, m and v separately)
            # find nodes to cross
            crossing_node_list = []
            if cur_node.node_type == 'material':
                if len(node_by_depth_m_extended[cur_depth+1]) > 0:
                    crossing_node_list = random.sample(node_by_depth_m_extended[cur_depth+1], min(_num_to_branch, len(node_by_depth_m_extended[cur_depth+1])))
            elif cur_node.node_type == 'group':
                if len(node_by_depth_v_extended[cur_depth+1]) > 0:
                    crossing_node_list = random.sample(node_by_depth_v_extended[cur_depth+1], min(_num_to_branch, len(node_by_depth_v_extended[cur_depth+1])))                
            else:
                raise ValueError('node type not recognized')
            # perform crossing
            for node_to_cross in crossing_node_list:
                PPNode.add_edge(parents=cur_node, children=node_to_cross)

            ## generating: create new nodes
            # adjustment for groups: at least two elements in a group
            if cur_node.node_type == 'group' and min(_num_to_cross, len(crossing_node_list)) + _num_to_branch <= 1:
                _num_to_branch = 2

            # generate new nodes
            for _ in range(_num_to_branch):
                # group node: virtual node seen as at the *same depth*
                if cur_node.node_type != 'group' and random.random() < sub_ratio[cur_depth + 1]:
                    new_node = PPNode(
                        id=get_funct_for_attr('name')('group', self.id_counters), 
                        node_type='group', 
                        depth=cur_depth
                    )
                    # 100% satisfy: depth<max_depth -> extend
                    dq.append(new_node)

                # normal material nodes:
                else:
                    new_node = PPNode(
                        id=get_funct_for_attr('name')('material', self.id_counters), 
                        node_type='material',
                        storage=get_funct_for_attr('storage')(sign=cur_node.node_type), 
                        depth=cur_depth + 1
                    )

                    # add to map
                    if cur_node.node_type == 'material':
                        node_by_depth_m[new_node.depth].append(new_node)
                    else:
                        node_by_depth_v[new_node.depth].append(new_node)

                    # satisfy: depth<max_depth -> extend
                    if new_node.depth < self.problem_size_config['max_depth']:
                        dq.append(new_node)

                # connect to parent
                PPNode.add_edge(parents=cur_node, children=new_node)

    def adding_orders(self, root, mapping_by_depth):
        """
        build four types of orders:
            1. normal ones: 1 order - 1 product
            2. middle products
            3. complete products
            4. shared products
        """

        ## init
        # create orders
        orders = get_funct_for_attr('orders')(order_num=self.problem_size_config['order_num'], name_counts=self.id_counters)

        ## get materials/products that corresponds to an order (orders primarily targets products; additional orders are allocated for materials)
        target_list = []
        if len(orders) <= len(root.children):
            # product num >= order num
            target_list = random.sample(root.children, len(orders))
        else:
            # product num < order num: adding middle products
            candidate_list = []
            for node_list_at_depth in mapping_by_depth:
                candidate_list.extend(node_list_at_depth)
            if len(candidate_list) >= len(orders) - len(root.children):
                target_list = root.children + random.sample(candidate_list, len(orders) - len(root.children))
            else:
                orders = orders[:len(root.children) + len(candidate_list)]

        ## concatenate
        random.shuffle(target_list)
        # disconnect all children of root
        for product_node in root.children:
            product_node.remove_parents(root)
        root.remove_all_children()

        # connect order-root, target-order
        for order_group, target in zip(orders, target_list):
            PPNode.add_edge(parents=root, children=order_group)
            PPNode.add_edge(parents=order_group, children=target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--product_num', type=int, default=800, help='Number of products')
    parser.add_argument('-o', '--order_num',type=int, default=2400, help='Number of orders')
    parser.add_argument('-e', '--end_num', type=int, default=300, help='End index of instance')
    parser.add_argument('-s', '--start_num', type=int, default=1, help='Start index of instance')
    parser.add_argument('-t', '--time', type=int, default=300, help='Time limit for MIP')
    args = parser.parse_args()
    generator = RandomGenerator(output_dir='data/raw', seed=5, start_num=args.start_num, end_num=args.end_num, product_num=args.product_num, order_num=args.order_num,
                                max_depth=5, data_type='train', time_limit=args.time)
    generator.run()
