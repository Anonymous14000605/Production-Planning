from collections import deque
import gurobipy as grb
import numpy as np


class MKNode:
    def __init__(self, name, v, node_type, depth=0):
        """
            nodes in the material kitting graph
        """

        # m, v: storage; p: demand
        self.storage = 0
        self.demand = 0
        if node_type == 'm':
            self.storage = v
        elif node_type == 'o':
            self.demand = v

        # stats
        self.name = name
        self.depth = depth
        self.children = []
        self.parents = []  # store [BomNode]
        self.visited = 0
        self.mip_var = {}  # store [SCIP_VARIABLE]
        self.consuming_rate = 1
        self.produce_ub = 0

        # type--  r_root, m_material, v_virtual-node
        self.node_type = node_type
        assert self.node_type in ['r', 'm', 'v', 'o']

    def reset(self):
        self.visited = 0
        self.mip_var = {}

    def add_child(self, new_edge):
        if type(new_edge) == list:
            self.children += new_edge
        else:
            self.children.append(new_edge)

    def add_parent(self, new_parent):
        if type(new_parent) == list:
            self.parents += new_parent
        else:
            self.parents.append(new_parent)

    def remove_child(self, child):
        self.children.remove(child)

    def remove_parent(self, parent):
        self.parents.remove(parent)


class MKGraph:
    def __init__(self, root):
        """
            graph-based representation of material kitting problem
        """

        self.root = None
        self.node_list = []
        self.root = root
        self.make_node_list()

    # reset all nodes' status
    def reset_nodes_stat(self):
        # reset status for every node
        # root not in node_list
        self.root.reset()
        for node in self.node_list:
            node.reset()

    # build self.node_list
    def make_node_list(self):
        self.node_list = []
        self.root.children.sort(key=lambda x: x.name)   # sort root's children by name dictionary order

        # build node_list
        dq = deque()
        dq.append(self.root)
        while dq:
            node = dq.pop()
            for c_node in node.children:
                if c_node.visited:  # check if visit for the first time
                    continue
                else:
                    c_node.children.sort(key=lambda x: x.name)  # sort c_node.children by name
                    self.node_list.append(c_node)
                    if c_node.children:
                        dq.append(c_node)

                    c_node.visited = 1  # update stat
        # reset 'visited'
        for node in self.node_list:
            node.visited = 0

        # update produce_ub
        for node in self.node_list:
            if node.node_type == 'm':
                node.produce_ub = np.inf
                for c in node.children:
                    node.produce_ub = min(node.produce_ub, c.storage // c.consuming_rate)


def build_mip(graph: MKGraph, save_dir):
    model = grb.Model()
    model.setParam('OutputFlag', 0)

    # add variables
    # add var
    # Notice: one node is visited once, as it appears once in the list
    for node in graph.node_list:
        if node.node_type == 'r':
            pass

        elif node.node_type == 'v':
            node.mip_var[node.name] = model.addVar(name=node.name, vtype=grb.GRB.INTEGER)
            # storage consumption
            node.mip_var[node.name + '_storage'] = model.addVar(name=node.name, vtype=grb.GRB.INTEGER)

        elif node.node_type == 'm':
            # requirement from parents
            for p_node in node.parents:
                node.mip_var[node.name + p_node.name] = \
                    model.addVar(name=node.name + p_node.name, vtype=grb.GRB.INTEGER)
            # production amount
            node.mip_var[node.name + '_production'] = \
                model.addVar(name=node.name, vtype=grb.GRB.INTEGER)
            # storage consumption
            node.mip_var[node.name + '_storage'] = \
                model.addVar(name=node.name, vtype=grb.GRB.INTEGER)

        elif node.node_type == 'o':
            node.mip_var[node.name] = model.addVar(name=node.name, vtype=grb.GRB.BINARY)

        else:
            raise ValueError('node type not supported')
    model.update()

    # add constraints
    for node in graph.node_list:
        if node.node_type == 'r':
            continue

        elif node.node_type == 'v':
            # production amount
            model.addConstr(
                grb.quicksum(c_node.mip_var[c_node.name + node.name] if c_node.node_type == 'm' else
                             c_node.mip_var[c_node.name] for c_node in node.children)
                + node.mip_var[node.name + '_storage']
                == node.mip_var[node.name]
            )

            # storage
            model.addConstr(node.mip_var[node.name + '_storage'] <= node.storage)

        elif node.node_type == 'o':
            # essential requirements
            model.addConstr(node.children[0].mip_var[node.children[0].name + node.name]
                            == node.demand * node.mip_var[node.name])

        elif node.node_type == 'm':
            # basic equations: sum(req_from(parent)) <= storage + production amount
            model.addConstr(grb.quicksum(node.mip_var[node.name + parent.name] for parent in node.parents)
                            * node.consuming_rate
                            ==
                            node.mip_var[node.name + '_production'] + node.mip_var[node.name + '_storage'])

            # essential requirements: req_to(child) = production amount
            for c_node in node.children:
                if c_node.node_type == 'm':
                    model.addConstr(c_node.mip_var[c_node.name + node.name] ==
                                    node.mip_var[node.name + '_production'])
                else:
                    model.addConstr(c_node.mip_var[c_node.name] == node.mip_var[node.name + '_production'])


            # produce upper bound
            model.addConstr(node.mip_var[node.name + '_production'] <= node.produce_ub)

            # storage upper bound
            model.addConstr(node.mip_var[node.name + '_storage'] <= node.storage)

        else:
            raise NotImplementedError

    # set objective
    # product num
    model.setObjective(grb.quicksum(product_node.mip_var[product_node.name] * product_node.demand
                                    for product_node in graph.root.children), sense=grb.GRB.MAXIMIZE)

    # save model
    model.write(save_dir)

